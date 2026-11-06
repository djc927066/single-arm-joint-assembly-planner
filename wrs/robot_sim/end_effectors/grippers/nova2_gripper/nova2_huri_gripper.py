import os
import math
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
from wrs.grasping.grasp import Grasp
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath


class Nova2HuriGripper(gpi.GripperInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT,
                 name="nova2_huri_gripper"):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)

        # flange
        cpl_end_pos = self.coupling.gl_flange_pose_list[0][0]
        cpl_end_rotmat = self.coupling.gl_flange_pose_list[0][1]

        self.jaw_range = np.array([0.0, 0.198])

        # jlc for base
        self.body = rkjlc.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, n_dof=1, name='base')
        self.body.jnts[0].loc_pos = np.array([0, 0, 0])
        self.body.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "base.stl"),
            name="nova2_huri_gripper_base",
            cdmesh_type=self.cdmesh_type,
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
            userdef_cdprim_fn=self._base_cdnp,
            ex_radius=.001)
        self.body.anchor.lnk_list[0].cmodel.rgba = np.array([0.57, 0.57, 0.57, 1])

        # jlc for left finger
        self.lft = rkjlc.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, n_dof=2, name='lft_finger')
        self.lft.jnts[0].loc_pos = np.array([-0.02507, -0.0272, 0.018595])
        self.lft.jnts[0].loc_rotmat = rm.rotmat_from_euler(0, 0, -math.pi)
        self.lft.jnts[0].loc_motion_ax = np.array([0, -1, 0])
        self.lft.jnts[0].motion_range = np.array([-math.pi, math.pi])
        self.lft.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "connector.stl"),
            name="lft_finger_connector",
            cdmesh_type=self.cdmesh_type,
            ex_radius=.001)
        self.lft.jnts[0].lnk.cmodel.rgba = np.array([0.65, 0.65, 0.65, 1])

        self.lft.jnts[1].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0.0, 0.099]))
        self.lft.jnts[1].loc_pos = np.array([-0.02507, -0.0272, 0.077905])
        self.lft.jnts[1].loc_motion_ax = np.array([-1, 0, 0])
        self.lft.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "finger.stl"),
            name="lft_finger_link",
            cdmesh_type=self.cdmesh_type,
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
            userdef_cdprim_fn=self._finger_cdnp,
            ex_radius=.001)
        self.lft.jnts[1].lnk.cmodel.rgba = np.array([0.65, 0.65, 0.65, 1])

        # jlc for right finger
        self.rgt = rkjlc.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, n_dof=2, name='rgt_finger')
        self.rgt.jnts[0].loc_pos = np.array([0.02507, 0.0272, 0.018595])
        self.rgt.jnts[0].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.rgt.jnts[0].loc_motion_ax = np.array([0, 1, 0])
        self.rgt.jnts[0].motion_range = np.array([-math.pi, math.pi])
        self.rgt.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "connector.stl"),
            name="rgt_finger_connector",
            cdmesh_type=self.cdmesh_type,
            ex_radius=.001)
        self.rgt.jnts[0].lnk.cmodel.rgba = np.array([0.65, 0.65, 0.65, 1])

        self.rgt.jnts[1].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0.0, 0.099]))
        self.rgt.jnts[1].loc_pos = np.array([-0.02507, -0.0272, 0.077905])
        self.rgt.jnts[1].loc_motion_ax = np.array([-1, 0, 0])
        self.rgt.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "finger.stl"),
            name="rgt_finger_link",
            cdmesh_type=self.cdmesh_type,
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
            userdef_cdprim_fn=self._finger_cdnp,
            ex_radius=.001)
        self.rgt.jnts[1].lnk.cmodel.rgba = np.array([0.65, 0.65, 0.65, 1])

        # finalize all jlchains
        self.body.finalize()
        self.lft.finalize()
        self.rgt.finalize()

        self.loc_acting_center_pos = np.array([0, 0, 0.225])
        self.loc_acting_center_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.cdelements = (self.body.anchor.lnk_list[0],
                           self.lft.jnts[0].lnk,
                           self.rgt.jnts[0].lnk,
                           self.lft.jnts[1].lnk,
                           self.rgt.jnts[1].lnk)

    @staticmethod
    def _finger_cdnp(name, ex_radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0.021 - 0.03, 0.002 + 0.0125, -0.02),
                                              x=0.03 + ex_radius, y=0.0125 + ex_radius, z=0.02 + ex_radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.0125, 0, 0.0125),
                                              x=0.0125 + ex_radius, y=0.025 + ex_radius, z=0.0125 + ex_radius)
        collision_node.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(0.011, 0, 0.0755),
                                              x=0.011 + ex_radius, y=0.015 + ex_radius, z=0.0725 + ex_radius)
        collision_node.addSolid(collision_primitive_c2)
        return NodePath(collision_node)

    @staticmethod
    def _base_cdnp(name, ex_radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0, 0, 0.04325),
                                              x=0.04 + ex_radius, y=0.0272 + ex_radius, z=0.04325 + ex_radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0, -0.006 - 0.026, 0.108),
                                              x=0.026 + ex_radius, y=0.026 + ex_radius, z=0.0115 + ex_radius)
        collision_node.addSolid(collision_primitive_c1)
        return NodePath(collision_node)

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.coupling.pos = self._pos
        self.coupling.rotmat = self._rotmat
        cpl_end_pos = self.coupling.gl_flange_pose_list[0][0]
        cpl_end_rotmat = self.coupling.gl_flange_pose_list[0][1]

        self.body.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.lft.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.update_oiee()

    def get_jaw_width(self):
        return -self.lft.jnts[1].motion_value * 2.0

    @gpi.ei.EEInterface.assert_oiee_decorator
    def change_jaw_width(self, jaw_width):
        side_jawwidth = jaw_width / 2.0
        if 0 <= jaw_width <= self.jaw_range[1]:
            self.lft.goto_given_conf(jnt_values=[0.0, -side_jawwidth])
            self.rgt.goto_given_conf(jnt_values=[0.0, -side_jawwidth])
        else:
            raise ValueError("The jaw_width parameter is out of range!")

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False):
        m_col = mmc.ModelCollection(name=self.name + '_stickmodel')
        self.coupling.gen_stickmodel(toggle_root_frame=False, toggle_flange_frame=False).attach_to(m_col)
        self.body.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        self.lft.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        self.rgt.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False):
        m_col = mmc.ModelCollection(name=self.name + '_meshmodel')
        self.coupling.gen_meshmodel(rgb=rgb,
                                    alpha=alpha,
                                    toggle_root_frame=False,
                                    toggle_flange_frame=False,
                                    toggle_cdmesh=toggle_cdmesh,
                                    toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.body.gen_meshmodel(rgb=rgb,
                                alpha=alpha,
                                toggle_flange_frame=False,
                                toggle_jnt_frames=toggle_jnt_frames,
                                toggle_cdmesh=toggle_cdmesh,
                                toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.lft.gen_meshmodel(rgb=rgb,
                               alpha=alpha,
                               toggle_flange_frame=False,
                               toggle_jnt_frames=toggle_jnt_frames,
                               toggle_cdmesh=toggle_cdmesh,
                               toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.rgt.gen_meshmodel(rgb=rgb,
                               alpha=alpha,
                               toggle_flange_frame=False,
                               toggle_jnt_frames=toggle_jnt_frames,
                               toggle_cdmesh=toggle_cdmesh,
                               toggle_cdprim=toggle_cdprim).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        # oiee
        self._gen_oiee_meshmodel(m_col, rgb=rgb, alpha=alpha, toggle_cdprim=toggle_cdprim,
                                 toggle_cdmesh=toggle_cdmesh, toggle_frame=toggle_jnt_frames)
        return m_col

    def get_grasp(self, ac_pos, ac_rotmat):
        """获取当前抓取姿势"""
        return Grasp(
            ee_values=self.get_jaw_width(),
            ac_pos=ac_pos,
            ac_rotmat=ac_rotmat
        )


if __name__ == '__main__':
    from wrs import wd, mgm

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    gripper = Nova2HuriGripper(cdmesh_type=mcm.const.CDMeshType.OBB)
    gripper.change_jaw_width(0.1)
    print(f"当前夹爪开口宽度: {gripper.get_jaw_width():.3f} m")
    gripper.gen_stickmodel().attach_to(base)
    gripper.gen_meshmodel(toggle_tcp_frame=True, toggle_jnt_frames=False, toggle_cdprim=False, alpha=1).attach_to(base)

    base.run()
