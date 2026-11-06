"""
Created on 2025/10/2 
Author: Hao Chen (chen960216@gmail.com)

Piper Arm Definition for WRS Simulator
-------------------------------------

This module defines a six‑degree‑of‑freedom (6‑DoF) model of the AgileX
PiPER robotic arm for use within the WRS simulator.  It closely follows
the structure of the ``Realman`` example supplied by the WRS framework,
but extracts geometric and kinematic parameters from the official PiPER
URDF specification.  Each joint’s local position, orientation,
actuation axis and motion limits are derived from the URDF file
``piper_description_v100_camera.urdf``.  Meshes
for the individual links are loaded from the same package and colours
are assigned to improve visualisation.

The resulting ``Piper`` class inherits from
``wrs.robot_sim.manipulators.manipulator_interface.ManipulatorInterface``
and can be instantiated and attached to a WRS ``World`` just like any
other manipulator.  For numerical inverse kinematics a ``trac_ik`` solver
may optionally be used when the library is available.

Links and joints defined:

1. **arm_base** → **link1** (joint1)
   * Translation: ``(0, 0, 0.123)``
   * Rotation: Euler angles ``(0, 0, −1.5708)``
   * Axis: ``(0,0,1)``
   * Range: ±150° (≈±2.618 rad)

2. **link1** → **link2** (joint2)
   * Translation: ``(0, 0, 0)`` m
   * Rotation: Euler angles ``(1.5708, −0.034907, −1.5708)`` rad
   * Axis: ``(0, 0, 1)``
   * Range: ``[0, π]`` rad

3. **link2** → **link3** (joint3)
   * Translation: ``(0.28358, 0.028726, 0)`` m
   * Rotation: Euler angles ``(0, 0, 0.06604341)`` rad
   * Axis: ``(0, 0, 1)``
   * Range: ``[−2.697, 0]`` rad

4. **link3** → **link4** (joint4)
   * Translation: ``(−0.24221, 0.068514, 0)`` m
   * Rotation: Euler angles ``(−1.5708, 0, 1.3826)`` rad
   * Axis: ``(0, 0, 1)``
   * Range: ±1.832 rad  

5. **link4** → **link5** (joint5)
   * Translation: ``(0, 0, 0)`` m
   * Rotation: Euler angles ``(1.5708, 0, 0)`` rad
   * Axis: ``(0, 0, 1)``
   * Range: ±1.22 rad   

6. **link5** → **link6** (joint6)
   * Translation: ``(0, 0.091, 0.0014165)`` m
   * Rotation: Euler angles ``(−1.5708, −1.5708, 0)`` rad
   * Axis: ``(0, 0, 1)``
   * Range: ±π rad    

Two prismatic joints (joint7 and joint8) and a camera link are defined
in the URDF but are omitted here because the WRS ``ManipulatorInterface``
currently supports only revolute joints.  Nevertheless, the attachment
point for tooling (TCP) can easily be adjusted via ``loc_tcp_pos``
and ``loc_tcp_rotmat``.
"""

import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface as mi
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd
import wrs.modeling.collision_model as mcm

try:
    from trac_ik import TracIK

    is_trac_ik = True
    print("Trac IK module loaded successfully")
except Exception as e:
    print(f"Trac IK module not loaded: {e}")
    is_trac_ik = False


class Piper(mi.ManipulatorInterface):
    """Model of the AgileX PiPER 6‑DoF arm for the WRS simulator."""

    def __init__(self, pos: np.ndarray = np.zeros(3), rotmat: np.ndarray = np.eye(3),
                 ik_solver: str = 'd', name: str = 'PiperArm', enable_cc: bool = False):
        """
        Initialise the PiPER arm.

        :param pos: World position of the arm base.
        :param rotmat: World orientation of the arm base.
        :param ik_solver: Either 'd' (default WRS numerical IK) or 'a'/'j'
                          to select alternative solvers.  When the
                          trac_ik library is available the TracIK solver
                          will be used.
        :param name: Identifier for this manipulator.
        :param enable_cc: Enable self‑collision checking if true.
        """
        super().__init__(pos=pos, rotmat=rotmat, home_conf=np.zeros(6),
                         name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)

        # define a uniform colour for all links
        rgba = np.array([0.6, 0.6, 0.6, 1.0])

        # anchor (arm_base)
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "base_link.STL"))
        # no rotation offset for the base link
        self.jlc.anchor.lnk_list[0].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rgba

        # --- Joint 1 (arm_base -> link1) ---
        self.jlc.jnts[0].loc_pos = np.array([0.0, 0.0, 0.123])
        # rotate about Z by −90° (−1.5708 rad)
        self.jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, -1.5708)
        self.jlc.jnts[0].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        self.jlc.jnts[0].motion_range = np.array([-2.618, 2.618])
        # assign the mesh for link1; orient it according to the URDF collision
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link1.STL"))
        self.jlc.jnts[0].lnk.loc_pos = np.array([0.0, 0.0, 0.0])
        # link1 in URDF has a rotation of +90° about Z for the collision mesh
        self.jlc.jnts[0].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 1.5708)
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1.0])

        # --- Joint 2 (link1 -> link2) ---
        self.jlc.jnts[1].loc_pos = np.array([0.0, 0.0, 0.0])
        # rotation sequence from URDF: roll=1.5708, pitch=-0.034907, yaw=-1.5708
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(1.5708, -0.034907, -1.5708)
        self.jlc.jnts[1].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        self.jlc.jnts[1].motion_range = np.array([-0.1, 3.14])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link2.STL"))
        self.jlc.jnts[1].lnk.loc_pos = np.array([0.0, 0.0, 0.0])
        # link2 has a small yaw offset of +0.1 rad for the collision mesh【338922380148493†L96-L100】
        self.jlc.jnts[1].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.1)
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1.0])

        # --- Joint 3 (link2 -> link3) ---
        self.jlc.jnts[2].loc_pos = np.array([0.28358, 0.028726, 0.0])
        # a small yaw rotation (≈3.78°)【338922380148493†L141-L147】
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.06604341)
        self.jlc.jnts[2].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        self.jlc.jnts[2].motion_range = np.array([-2.697, 0.0])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link3.STL"))
        self.jlc.jnts[2].lnk.loc_pos = np.array([0.0, 0.0, 0.0])
        # link3 has a yaw offset of −1.75 rad on the collision mesh【338922380148493†L134-L137】
        self.jlc.jnts[2].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, -1.75)
        self.jlc.jnts[2].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1.0])

        # --- Joint 4 (link3 -> link4) ---
        self.jlc.jnts[3].loc_pos = np.array([-0.24221, 0.068514, 0.0])
        # rotation: roll=-90°, yaw≈1.3826 rad  
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(-1.5708, 0.0, 1.3826)
        self.jlc.jnts[3].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        self.jlc.jnts[3].motion_range = np.array([-1.832, 1.832])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link4.STL"))
        self.jlc.jnts[3].lnk.loc_pos = np.array([0.0, 0.0, 0.0])
        # link4 has no additional rotation
        self.jlc.jnts[3].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, np.pi)
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1.0])

        # --- Joint 5 (link4 -> link5) ---
        self.jlc.jnts[4].loc_pos = np.array([0.0, 0.0, 0.0])
        # rotation: roll=90°   
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(1.5708, 0.0, 0.0)
        self.jlc.jnts[4].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        self.jlc.jnts[4].motion_range = np.array([-1.22, 1.22])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link5.STL"))
        self.jlc.jnts[4].lnk.loc_pos = np.array([0.0, 0.0, 0.0])
        # link5 collision mesh is rotated by −π about Z【338922380148493†L210-L213】
        self.jlc.jnts[4].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, -3.14)
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1.0])

        # --- Joint 6 (link5 -> link6) ---
        self.jlc.jnts[5].loc_pos = np.array([0.0, 0.091, 0.0014165])
        # orig_rotmat = rm.rotmat_from_euler(-1.5708, -1.5708, 0.0)
        corrected_rotmat_direct = rm.rotmat_from_euler(1.5708, 0.0, 3.14159)
        self.jlc.jnts[5].loc_rotmat = corrected_rotmat_direct
        self.jlc.jnts[5].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        self.jlc.jnts[5].motion_range = np.array([-2.094, 2.094])
        # self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(
        #     os.path.join(current_file_dir, "meshes", "camera_v3.dae"))
        # In the URDF the camera (link6) mesh has an offset
        # self.jlc.jnts[5].lnk.loc_pos = np.array([-0.002, -0.008, 0.0])
        # self.jlc.jnts[5].lnk.loc_rotmat = rm.rotmat_from_euler(-1.57, 0.0, 0.0)
        # self.jlc.jnts[5].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1.0])

        # Finalise the joint linkage chain and set up IK solver
        self.jlc.finalize(ik_solver=ik_solver, identifier_str=name)
        self.loc_tcp_pos = np.array([0.0, 0.0, 0.0])
        self.loc_tcp_rotmat = np.eye(3)
        # z_rot_90 = rm.rotmat_from_euler(0.0, 0.0, -np.pi / 2)
        #
        # # 右乘操作 (A = A @ B) 实现绕局部坐标系旋转
        # self.loc_tcp_rotmat = self.loc_tcp_rotmat @ z_rot_90
        # Configure the Trac IK solver when available
        if is_trac_ik:
            directory = os.path.abspath(os.path.dirname(__file__))
            urdf = os.path.join(directory, "piper_description_v100_camera.urdf")
            # base_link and link6 are defined in the URDF; use them for IK
            self._ik_solver = TracIK("arm_base", "link6", urdf,
                                     timeout=0.002,
                                     solver_type="Distance")
        else:
            self._ik_solver = None

        # ── TracIK 抗抽风：多 seed 重试配置（默认关闭）───────
        # TracIK 内部用随机种子做 SQP/Newton 迭代，timeout 内只抽一次，
        # 临近关节限位的目标偶尔 return None。下面提供"显式启用"的
        # 多 seed 重试机制：``ik()`` 主调用失败时按 home_conf -> N 个
        # 随机 seed 多试几次，任一成功即返回。
        #
        # 默认 ``_ik_retry_n = 0`` —— 行为与原代码完全一致，避免给
        # ``reason_common_gids`` 这类批量 IK 调用引入 5-10x 失败延迟。
        # 调用方（如 fast_layout_search）想要抗抽风时显式设置：
        #     arm._ik_retry_n = 4
        # 失败成本 ≈ retry_n * timeout（仅当 trac_ik 主调用 None 时）。
        self._ik_retry_n = 0
        self._ik_rng = np.random.default_rng(0)

        # Set up collision checking (self‑collision) if requested
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self) -> None:
        """Configure pairs of links for self‑collision checking."""
        # Add each link to the collision checker and establish
        # conservative collision pairs (into and from lists).  The
        # selections below are similar to those used in the Realman
        # example.
        lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])  # base
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        from_list = [l3,l4]
        into_list = [ l0, l1]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def ik(self, tgt_pos: np.ndarray, tgt_rotmat: np.ndarray,
           seed_jnt_values=None, option: str = "empty", toggle_dbg: bool = False):
        """
        Solve the inverse kinematics for the end‑effector.

        When the Trac IK library is available this method delegates to
        ``TracIK.ik``; otherwise it falls back on the numerical IK
        implementation provided by the JLC.  An optional seed
        configuration may be provided to bias the solution.

        :param tgt_pos: Desired TCP position (3×1 array).
        :param tgt_rotmat: Desired TCP rotation matrix (3×3).
        :param seed_jnt_values: Optional initial joint values (6×1 array).
        :param option: Additional option string (unused here).
        :param toggle_dbg: If true prints debugging information.
        :return: Joint configuration achieving the target pose or None
        """
        # Transform target pose into the wrist coordinate frame
        tgt_rotmat = tgt_rotmat @ self.loc_tcp_rotmat.T
        tgt_pos = tgt_pos - tgt_rotmat @ self.loc_tcp_pos

        if is_trac_ik and self._ik_solver is not None:
            # convert to the base_link frame for Trac IK
            anchor_inv_homomat = np.linalg.inv(rm.homomat_from_posrot(
                self.jlc.anchor.pos, self.jlc.anchor.rotmat))
            tgt_homomat = anchor_inv_homomat.dot(rm.homomat_from_posrot(tgt_pos, tgt_rotmat))
            tgt_pos, tgt_rotmat = tgt_homomat[:3, 3], tgt_homomat[:3, :3]
            # —— 主调用：用调用方传的 seed（或 home_conf）—————————
            primary_seed = (self.home_conf if seed_jnt_values is None
                            else np.asarray(seed_jnt_values).copy())
            result = self._ik_solver.ik(tgt_pos, tgt_rotmat,
                                         seed_jnt_values=primary_seed)
            if result is not None:
                return result
            # —— 抽风兜底：仅当显式启用 (_ik_retry_n > 0) 时多 seed 重试 ──
            # 顺序：home_conf（若主 seed 不是它）→ N 个随机关节值
            # 任一成功立即返回；全部失败仍 return None（与原行为兼容）。
            retry_n = int(getattr(self, "_ik_retry_n", 0))
            if retry_n <= 0:
                return None
            tried_home = bool(np.allclose(primary_seed, self.home_conf))
            if not tried_home:
                result = self._ik_solver.ik(
                    tgt_pos, tgt_rotmat,
                    seed_jnt_values=self.home_conf.copy())
                if result is not None:
                    return result
            jr = self.jnt_ranges
            for _ in range(retry_n):
                rand_seed = self._ik_rng.uniform(jr[:, 0], jr[:, 1])
                result = self._ik_solver.ik(
                    tgt_pos, tgt_rotmat, seed_jnt_values=rand_seed)
                if result is not None:
                    return result
            return None
        else:
            # fall back to numerical IK provided by the JLC
            return self.jlc.ik(tgt_pos=tgt_pos,
                               tgt_rotmat=tgt_rotmat,
                               seed_jnt_values=seed_jnt_values,
                               toggle_dbg=toggle_dbg)


# if __name__ == '__main__':
#     import wrs.visualization.panda.world as wd
#
#     base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
#     arm = Piper()
#     # arm.gen_meshmodel().attach_to(base)
#     mgm.gen_frame().attach_to(base)
#
#     tgt_pos = np.array([0.378, -0.099417, 0.157612])
#     tgt_rotmat = rm.rotmat_from_euler(3.0369, -0.0483, 2.7970)
#     mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
#     jnt_values = arm.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, toggle_dbg=False)
#     print(jnt_values)
#     if jnt_values is not None:
#         arm.goto_given_conf(jnt_values=jnt_values)
#         arm.gen_meshmodel(alpha=1, toggle_tcp_frame=True).attach_to(base)
#     else:
#         print(1111)
#     arm.gen_meshmodel().attach_to(base)
#     # arm.show_cdprim()
#     base.run()
if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    arm = Piper(enable_cc = True)
    # arm.goto_given_conf(np.array([ 0.02003638,  1.81482826, -1.24311076, -0.06082472,  1.16155152,
    #    -0.20125392]))
    arm.goto_given_conf(np.array([0,0,0,0,0,0]))
    arm.gen_meshmodel().attach_to(base)
    mcm.gen_box(xyz_lengths=np.array([0.5, 0.6, 0.03]),
                                       pos=rm.vec(0, 0, 0.6),
                                       rotmat=np.eye(3),
                                       rgb=[0.1, 0.1, 0.2],
                                       alpha=0.8).attach_to(base)

    mcm.gen_box(xyz_lengths=np.array([0.5, 0.6, 0.06]),
                pos=rm.vec(0, 0, -0.04),
                rotmat=np.eye(3),
                rgb=[0.1, 0.1, 0.2],
                alpha=0.8).attach_to(base)

    print(arm.is_collided())
    # arm.show_cdprim()
    arm.gen_stickmodel(toggle_jnt_frames=True,toggle_tcp_frame=True).attach_to(base)
    base.run()
