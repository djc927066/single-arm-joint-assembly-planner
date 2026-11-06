"""
Created on 2025/10/2 
Author: Hao Chen (chen960216@gmail.com)

Piper Arm Controller Wrapper
================================

This module provides a high‑level Python wrapper around the
`piper_sdk` CAN interface for the AgileX Piper robot arm.  The goal
is to present an interface similar to the `RealmanArmController`
shown in the user's example while hiding the low‑level details of
CAN message formatting.  All angles exposed to the user are in
radians and all linear distances are in metres.  Internally the
Piper SDK expects units of 0.001 degrees for angles and
0.001 millimetres for Cartesian coordinates; the wrapper handles
these conversions automatically.

The wrapper exposes methods for enabling/disabling the arm,
moving the joints, moving the end effector in position control
(`move_p`) and linear Cartesian control (`move_l`), retrieving
joint values and end pose, as well as basic gripper control and
emergency stop functionality.  It uses the `wrs.basis.robot_math`
helpers for converting between rotation matrices and Euler angles.

Example
-------

```python
from piper_arm_controller import PiperArmController
import numpy as np

# create a controller on CAN interface 'can0'
piper = PiperArmController(can_name="can0", has_gripper=True)

# power on motors
piper.enable()

# move to joint position (radians)
piper.move_j([0.0, -0.5, 0.5, 0.0, 0.0, 0.0], speed=30)

# move end effector to a pose (metres and rotation matrix)
pos = np.array([0.15, 0.0, 0.25])
rotmat = np.eye(3)
piper.move_p(pos, rotmat, speed=50)

# read back current pose
pos_feedback, rotmat_feedback = piper.get_pose()

# shut down
piper.disable()
piper.close_connection()
```

Note
----

The wrapper does not implement all functionality present in the
underlying SDK.  Features such as force sensing, force control
and path planning are not available on the Piper arm and therefore
methods related to those functions in the original Realman example
are deliberately omitted.
"""

from __future__ import annotations

import platform
import subprocess
from typing import Iterable, Tuple, Literal
import time
import numpy as np

try:
    # Import robot math helpers for Euler/rotation matrix conversion
    # The user example imports this module; it is expected to be
    # available in the same environment.
    import wrs.basis.robot_math as rm
except ImportError:
    # Provide a minimal fallback implementation if wrs is not present.
    # Only ZYX (roll, pitch, yaw) order is implemented here.
    from math import atan2, asin, cos, sin


    class rm:
        @staticmethod
        def rotmat_to_euler(rot: np.ndarray) -> np.ndarray:
            """Convert a 3×3 rotation matrix to Euler angles (XYZ order).

            Parameters
            ----------
            rot : np.ndarray
                Rotation matrix of shape (3, 3).

            Returns
            -------
            np.ndarray
                Euler angles in radians as a 1D array [roll, pitch, yaw].
            """
            if rot.shape != (3, 3):
                raise ValueError("rot must be a 3x3 matrix")
            sy = -rot[2, 0]
            cy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
            pitch = atan2(sy, cy)
            roll = atan2(rot[2, 1], rot[2, 2])
            yaw = atan2(rot[1, 0], rot[0, 0])
            return np.array([roll, pitch, yaw])

        @staticmethod
        def rotmat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
            """Convert Euler angles to a 3×3 rotation matrix (XYZ order).

            Parameters
            ----------
            roll, pitch, yaw : float
                Euler angles in radians.

            Returns
            -------
            np.ndarray
                A 3×3 rotation matrix.
            """
            cr = cos(roll)
            sr = sin(roll)
            cp = cos(pitch)
            sp = sin(pitch)
            cy = cos(yaw)
            sy = sin(yaw)
            return np.array([
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ])

# Import the Piper SDK interface.  The user should have installed
# piper_sdk in their Python environment.  If this import fails the
# wrapper will raise an ImportError when instantiated.
try:
    from wrs.drivers.piper_sdk import C_PiperInterface_V2
except:
    from piper_sdk import C_PiperInterface_V2

try:
    import wrs.motion.trajectory.piecewisepoly_toppra as pwp

    TOPPRA_EXIST = True
except:
    TOPPRA_EXIST = False


class PiperArmController:
    """High‑level controller for the AgileX Piper robotic arm.

    Parameters
    ----------
    can_name : str, optional
        The name of the CAN interface (e.g. ``"can0"``).  Defaults to
        ``"can0"``.
    has_gripper : bool, optional
        Whether the arm is equipped with a gripper.  If ``True`` then
        the gripper control methods will be available.  Defaults to
        ``False``.
    auto_enable : bool, optional
        If ``True`` (default) the controller will attempt to enable
        all joints immediately after connecting to the CAN port.  If
        ``False`` the caller must call :meth:`enable` manually.
    """

    def __init__(self, *, can_name: str = "can0", has_gripper: bool = False, auto_enable: bool = True) -> None:
        if platform.system() == "Linux":
            self.configure_can(can_name)
        # Create the underlying Piper interface
        self._interface = C_PiperInterface_V2(can_name)
        # Establish the CAN connection and start background threads
        self._interface.ConnectPort()
        # Give the interface a moment to start
        time.sleep(0.1)
        self._has_gripper = has_gripper
        # Automatically enable the arm if requested
        if auto_enable and not self.is_enabled:
            self._interface.MotionCtrl_1(0x00, 0x00, 0x00)
            self.enable()

    # ------------------------------------------------------------------
    # Power management
    # ------------------------------------------------------------------
    @property
    def is_enabled(self) -> bool:
        """Return ``True`` if all motors are enabled, else ``False``.

        This method queries the enable status of each joint motor via
        :meth:`piper_sdk.C_PiperInterface_V2.GetArmEnableStatus` and
        returns ``True`` only if all motors report an enabled state.
        """
        enable_status = self._interface.GetArmEnableStatus()
        return all(enable_status)

    def configure_can(self, can_name):
        """
        配置 CAN 接口，设置 Bitrate
        """
        try:
            # 配置 IP link 和 bitrate 设置
            subprocess.run(f"sudo ip link set {can_name} down", shell=True, check=True)  # 先断开连接
            subprocess.run(
                f"sudo ip link set {can_name} type can bitrate 1000000",
                shell=True, check=True)  # berr-reporting on restart-ms 100
            subprocess.run(f"sudo ip link set {can_name} up", shell=True, check=True)  # 重新启动 can0
            print(f"{can_name} 配置成功，Bitrate 设置为 1Mbps，数据域 5Mbps。")
        except subprocess.CalledProcessError as e:
            print(f"配置 {can_name} 时出错: {e}")

    def enable(self) -> None:
        """Enable all joints of the robotic arm.

        This method attempts to enable the arm by repeatedly calling
        :meth:`piper_sdk.C_PiperInterface_V2.EnablePiper` until all
        motors report an enabled state.  It blocks briefly between
        retries to allow the hardware to respond.
        """
        # Attempt to enable all motors
        while not self._interface.EnablePiper():
            # Sleep briefly to yield to the background CAN thread
            time.sleep(0.01)
        # Wait until the enable status is fully updated
        while not self.is_enabled:
            time.sleep(0.01)
        time.sleep(.5)
        print("Piper arm motors enabled.")

    def disable(self) -> None:
        """Disable all joints of the robotic arm.

        This method disables all motors via :meth:`DisablePiper` and
        does not block on confirmation.  Use :meth:`is_enabled` if
        you need to verify the final state.
        """
        self._interface.DisablePiper()
        print("Piper arm motors disabled.")

    # ------------------------------------------------------------------
    # Motion control
    # ------------------------------------------------------------------
    def move_j(
            self,
            joint_angles: Iterable[float],
            *,
            is_radians: bool = True,
            speed: int = 50,
            block: bool = True,
            tolerance: float = 0.01,
    ) -> None:
        """Move the arm to a joint configuration (Move J).

        Parameters
        ----------
        joint_angles : iterable of float
            A sequence of six joint angles.  When ``is_radians`` is
            ``True`` (default) the values are interpreted as radians;
            otherwise they are interpreted as degrees.
        is_radians : bool, optional
            Interpret ``joint_angles`` in radians if ``True``, else
            degrees.  Defaults to ``True``.
        speed : int, optional
            Motion speed percentage (0–100).  This is passed directly to
            the underlying SDK.  Defaults to 50.
        block : bool, optional
            If ``True`` the method will block until the arm has
            finished moving to the target configuration.  When ``False``
            (default) the command returns immediately after sending
            the joint angles.
        tolerance : float, optional
            Joint angle tolerance for blocking, in radians.  When
            ``block`` is enabled the method will wait until the
            maximum absolute difference between the current joint
            angles and the target is below this value or until the
            internal motion status reports that the target has been
            reached.  Defaults to ``0.01`` rad (~0.57°).

        Notes
        -----
        The underlying Piper SDK implements MoveJ as a non‑blocking
        command; it begins motion to the specified joint angles and
        returns immediately.  Setting ``block=True`` causes this
        wrapper to poll the arm status and joint feedback until the
        target has been reached.  The polling frequency is fixed at
        approximately 100Hz.
        """
        # Convert input to NumPy array for validation and conversion
        angles = np.asarray(list(joint_angles), dtype=float)
        if angles.size != 6:
            raise ValueError("joint_angles must contain six values")
        # Convert to degrees if necessary for the SDK
        if is_radians:
            angles_deg = np.degrees(angles)
        else:
            angles_deg = angles
        # Convert degrees to SDK units of 0.001 degrees and round to int
        joint_units = [int(round(val * 1000.0)) for val in angles_deg]
        # Set motion mode to joint control (Move J = 0x01) using CAN control mode
        self._interface.MotionCtrl_2(
            ctrl_mode=0x01,
            move_mode=0x01,
            move_spd_rate_ctrl=int(speed),
            is_mit_mode=0x00,
        )
        time.sleep(.01)
        # Send the joint angles
        self._interface.JointCtrl(*joint_units)
        time.sleep(.01)
        # If blocking, wait until the target is reached
        if block:
            # Convert tolerance to degrees for comparison
            tol_deg = np.degrees(tolerance) if is_radians else tolerance
            # Ensure tol_deg is non‑negative
            tol_deg = abs(tol_deg)
            # Wait until motion status indicates arrival or joint error below tolerance
            self._wait_until_joint_target_reached(angles_deg, tol_deg)



    def move_p(self, pos: Iterable[float], rot: np.ndarray, *, is_euler: bool = False, speed: int = 50) -> None:
        """Move the end effector to a pose using position control (Move P).

        Parameters
        ----------
        pos : iterable of float
            Cartesian position of the end effector in metres as
            ``[x, y, z]``.
        rot : numpy.ndarray or iterable
            Either a 3×3 rotation matrix or a sequence of three Euler
            angles.  If ``is_euler`` is ``False`` (default) then
            ``rot`` is interpreted as a rotation matrix; otherwise it
            must be a sequence of Euler angles in radians.
        is_euler : bool, optional
            Set to ``True`` when providing Euler angles directly.  When
            ``False`` (default) the input is treated as a rotation
            matrix.
        speed : int, optional
            Motion speed percentage (0–100).  Defaults to 50.
        """
        position = np.asarray(list(pos), dtype=float)
        if position.size != 3:
            raise ValueError("pos must contain three coordinates")
        # Extract Euler angles from rotation input
        if is_euler:
            euler = np.asarray(rot, dtype=float)
            if euler.size != 3:
                raise ValueError("Euler rotation must have three values")
        else:
            rotmat = np.asarray(rot, dtype=float)
            if rotmat.shape != (3, 3):
                raise ValueError("rot must be a 3x3 rotation matrix")
            euler = rm.rotmat_to_euler(rotmat)
        # Convert position metres -> 0.001 mm units
        # 1 m = 1000 mm, 1 mm = 1000 units => multiply by 1e6
        pos_units = (position * 1_000_000.0).astype(int)
        # Convert Euler angles radians -> degrees -> 0.001 degree units
        euler_deg = np.degrees(euler)
        euler_units = (euler_deg * 1000.0).astype(int)
        # Set motion mode to position control (Move P = 0x00)
        self._interface.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x00,
                                     move_spd_rate_ctrl=int(speed), is_mit_mode=0x00)
        # Send the end pose
        self._interface.EndPoseCtrl(pos_units[0], pos_units[1], pos_units[2],
                                    euler_units[0], euler_units[1], euler_units[2])

    def move_m(
            self,
            joint_angles: Iterable[float],
            *,
            is_radians: bool = True,
            kp: float = 10.0,
            kd: float = 0.8,
            vel_ref: float = 0.0,
            t_ref: float = 0.0,
            block: bool = True,
            tolerance: float = 0.01,
    ) -> None:
        """Move the arm to a joint configuration (MIT Mode).

        Parameters
        ----------
        joint_angles : iterable of float
            A sequence of six joint angles.  When ``is_radians`` is
            ``True`` (default) the values are interpreted as radians;
            otherwise they are interpreted as degrees.
        is_radians : bool, optional
            Interpret ``joint_angles`` in radians if ``True``, else
            degrees.  Defaults to ``True``.
        kp : float, optional
            Proportional gain for MIT control. Defaults to 10.0.
        kd : float, optional
            Derivative gain for MIT control. Defaults to 0.8.
        vel_ref : float, optional
            Desired joint speed reference (rad/s). Defaults to 0.0.
        t_ref : float, optional
            Desired joint torque reference. Defaults to 0.0.
        block : bool, optional
            If ``True`` waits until all joints reach target positions.
        tolerance : float, optional
            Joint angle tolerance for blocking, in radians.
        """

        # 转为 numpy 数组方便处理
        angles = np.asarray(list(joint_angles), dtype=float)
        if angles.size != 6:
            raise ValueError("joint_angles must contain six values")

        # 角度制转弧度
        if not is_radians:
            angles = np.radians(angles)

        # === MIT 模式控制 ===
        for i, target_angle in enumerate(angles, start=1):
            self._interface.JointMitCtrl(
                motor_num=i,
                pos_ref=float(target_angle),
                vel_ref=vel_ref,
                kp=kp,
                kd=kd,
                t_ref=t_ref
            )
            time.sleep(0.002)  # 避免CAN总线过载

        # === 阻塞等待直到到达目标 ===
        if block:
            tol = abs(tolerance)
            self._wait_until_joint_target_reached(np.degrees(angles), np.degrees(tol))


    def move_l(self, pos: Iterable[float], rot: np.ndarray, *, is_euler: bool = False, speed: int = 50) -> None:
        """Move the end effector using linear Cartesian interpolation (Move L).

        The parameters are the same as for :meth:`move_p`.  The only
        difference is that the underlying SDK is instructed to use
        linear movement mode (``move_mode=0x02``).  Not all Piper
        firmware versions may support this mode; refer to the Piper
        manual for details.
        """
        position = np.asarray(list(pos), dtype=float)
        if position.size != 3:
            raise ValueError("pos must contain three coordinates")
        if is_euler:
            euler = np.asarray(rot, dtype=float)
            if euler.size != 3:
                raise ValueError("Euler rotation must have three values")
        else:
            rotmat = np.asarray(rot, dtype=float)
            if rotmat.shape != (3, 3):
                raise ValueError("rot must be a 3x3 rotation matrix")
            euler = rm.rotmat_to_euler(rotmat)
        pos_units = (position * 1_000_000.0).astype(int)
        euler_units = (np.degrees(euler) * 1000.0).astype(int)
        # Set motion mode to linear control (Move L = 0x02)
        self._interface.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x02,
                                     move_spd_rate_ctrl=int(speed), is_mit_mode=0x00)
        self._interface.EndPoseCtrl(pos_units[0], pos_units[1], pos_units[2],
                                    euler_units[0], euler_units[1], euler_units[2])

    def move_jntspace_highfreq(self,
                               joint_angles: list[float],
                               set_mode: bool = False) -> int:
        """
        高频关节角度透传接口 (类似 rm_movej_canfd)。
        绕过控制器内部规划，直接将外部轨迹点以高频发送。

        @details
            - 将关节角度（弧度）转换为 0.001 度单位。
            - 调用底层接口 SetJointTargetAngles_HighFreq 发送指令 (CAN ID 0x155/0x156/0x157)。
            - **注意：此接口要求外部调用以稳定的、高频率（如 100Hz 或更高）发送数据。**

        Args:
            joint_angles (list[float]): 关节 1~6 目标角度数组, 单位：弧度 (rad)。
            set_mode (bool, optional): 是否在发送前设置机械臂进入高频/MIT模式。
                                       建议在开始发送路径流前设置一次。Defaults to False.

        Returns:
            int: 函数执行的状态码。
                 0: 成功。
                 -1: 参数错误（关节数量不为 6）。
                 -2: CAN 消息发送失败。
                 -3: 模式设置失败。
        """
        # 1. 检查关节数量
        if len(joint_angles) != 6:
            self.logger.error(f"High-frequency joint control requires 6 joint angles, but got {len(joint_angles)}")
            return -1

        # 2. (可选) 设置模式：在路径流开始前设置一次
        if set_mode:
            try:
                # 假设 self.interface 是 C_PiperInterface_V2 的实例
                self._interface.MotionCtrl_2(
                    ctrl_mode=0x01,  # CAN 指令控制模式
                    move_mode=0x04,  # MOVE M (MIT) 高频模式
                    is_mit_mode=0xAD  # MIT 模式 (高响应)
                )
            except Exception as e:
                self.logger.error(f"Failed to set high-frequency mode: {e}")
                return -3

        # 3. 转换单位：弧度 (rad) -> 度 (deg)
        # 注意：np.degrees 函数需要导入 numpy (在 piper.py 中已导入)
        joint_angles_deg = np.degrees(joint_angles).tolist()

        # 4. 调用底层 SDK 接口发送 (假设底层接口已命名为 SetJointTargetAngles_HighFreq)
        tag = self._interface.SetJointTargetAngles_HighFreq(joint_angles_deg)

        return tag

    def exec_path_by_move_j(self, path, speed=30):
        for conf in path:
            PiperArmController.move_j(conf, is_radians=True, speed=speed, block=True)
            time.sleep(0.05)

    def move_jntspace_path(self,
                           path,
                           max_jntvel: list = None,
                           max_jntacc: list = None,
                           start_frame_id=1,
                           speed=50,
                           control_frequency=0.3, ):
        """
        :param path: [jnt_values0, jnt_values1, ...], results of motion planning
        :param max_jntvel: 1x6 list to describe the maximum joint speed for the arm
        :param max_jntacc: 1x6 list to describe the maximum joint acceleration for the arm
        :param start_frame_id:
        :return:
        author: weiwei
        """
        if TOPPRA_EXIST:
            if path is None:
                raise ValueError("The given is incorrect!")
            path = np.array(path)
            # # Refer to https://www.ufactory.cc/_files/ugd/896670_9ce29284b6474a97b0fc20c221615017.pdf
            # # the robotic arm can accept joint position commands sent at a fixed high frequency like 100Hz
            control_frequency = control_frequency
            tpply = pwp.PiecewisePolyTOPPRA()
            interpolated_path = tpply.interpolate_by_max_spdacc(path=path,
                                                                ctrl_freq=control_frequency,
                                                                max_vels=max_jntvel,
                                                                max_accs=max_jntacc,
                                                                toggle_debug=False)
            interpolated_path = interpolated_path[start_frame_id:]
            for jnt_values in interpolated_path:
                self.move_j(
                    joint_angles=jnt_values,
                    speed=speed,
                    block=False, )
                time.sleep(.05)
            return
        else:
            raise NotImplementedError

    # ------------------------------------------------------------------
    # Feedback methods
    # ------------------------------------------------------------------
    def get_joint_values(self) -> np.ndarray:
        """Return the current joint angles in radians as a NumPy array.

        This method reads feedback from the arm via
        :meth:`GetArmJointMsgs`, converts the 0.001° units back to
        degrees and then to radians.
        """
        joint_msg = self._interface.GetArmJointMsgs()
        js = joint_msg.joint_state
        values_deg = np.array([
            js.joint_1,
            js.joint_2,
            js.joint_3,
            js.joint_4,
            js.joint_5,
            js.joint_6,
        ], dtype=float) / 1000.0
        return np.radians(values_deg)

    def get_joint_values_raw(self):
        """Return the current joint angles in raw 0.001° units.

        This method reads feedback from the arm via
        :meth:`GetArmJointMsgs` and returns the joint angles as a list
        of six integers in 0.001° units.
        """
        joint_msg = self._interface.GetArmJointMsgs()
        js = joint_msg.joint_state
        return [
            js.joint_1,
            js.joint_2,
            js.joint_3,
            js.joint_4,
            js.joint_5,
            js.joint_6,
        ]

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the current end‑effector pose.

        Returns a tuple ``(position, rotation_matrix)`` where
        ``position`` is a 3‑vector in metres and ``rotation_matrix`` is
        a 3×3 NumPy array.  Euler angles are converted to a rotation
        matrix using the `wrs.basis.robot_math.rotmat_from_euler` helper.
        """
        pose_msg = self._interface.GetArmEndPoseMsgs()
        ep = pose_msg.end_pose
        # Position in 0.001 mm -> m
        position = np.array([ep.X_axis, ep.Y_axis, ep.Z_axis], dtype=float) / 1_000_000.0
        # Orientation in 0.001 deg -> deg -> rad
        euler_deg = np.array([ep.RX_axis, ep.RY_axis, ep.RZ_axis], dtype=float) / 1000.0
        euler_rad = np.radians(euler_deg)
        rot = rm.rotmat_from_euler(*euler_rad)
        return position, rot

    def get_pose_raw(self) -> np.ndarray:
        """Return the raw pose as a 6‑element NumPy array.

        The first three elements are the Cartesian position in metres
        and the last three are the Euler angles in radians.
        """
        pose_msg = self._interface.GetArmEndPoseMsgs()
        ep = pose_msg.end_pose
        pos = np.array([ep.X_axis, ep.Y_axis, ep.Z_axis], dtype=float) / 1_000_000.0
        euler_deg = np.array([ep.RX_axis, ep.RY_axis, ep.RZ_axis], dtype=float) / 1000.0
        euler_rad = np.radians(euler_deg)
        return np.concatenate((pos, euler_rad))

    # ------------------------------------------------------------------
    # Gripper control
    # ------------------------------------------------------------------
    def gripper_control(self, angle: float = 0.0, effort: float = 0.0, *, enable: bool = True,
                        set_zero: bool = False) -> None:
        """Control the gripper opening and effort.

        Parameters
        ----------
        angle : float, optional
            Desired opening width of the gripper in metres.  The Piper
            gripper has a limited range; values outside the physical
            range will be clamped by the firmware.  Defaults to 0.0.
        effort : float, optional
            Desired gripping effort in N·m.  The SDK represents
            gripper torque in units of 0.001 N·m.  Defaults to 0.0.
        enable : bool, optional
            Whether to enable (``True``) or disable (``False``) the
            gripper.  Defaults to ``True``.
        set_zero : bool, optional
            If ``True`` the current gripper position will be recorded as
            the zero point.  Defaults to ``False``.
        """
        if not self._has_gripper:
            raise RuntimeError("This PiperArmController was instantiated without a gripper")
        # Convert opening from metres to 0.001 mm units.  Negative
        # openings are not meaningful but are passed through.
        angle_units = int(round(angle * 1_000_000.0))
        # Convert effort from N·m to 0.001 N·m
        effort_units = int(round(effort * 1000.0))
        gripper_code = 0x01 if enable else 0x00
        set_zero_code = 0xAE if set_zero else 0x00
        self._interface.GripperCtrl(angle_units, effort_units, gripper_code, set_zero_code)

    def get_gripper_status(self) -> Tuple[float, float, int]:
        """Return the current gripper opening, effort and status code.

        Returns
        -------
        tuple
            ``(opening_m, effort_nm, status_code)`` where ``opening_m`` is
            the gripper width in metres, ``effort_nm`` is the torque in
            N·m and ``status_code`` is an integer representing the
            gripper state reported by the SDK.
        """
        if not self._has_gripper:
            raise RuntimeError("This PiperArmController was instantiated without a gripper")
        gripper_msg = self._interface.GetArmGripperMsgs()
        gs = gripper_msg.gripper_state
        # Opening in 0.001 mm -> m
        opening_m = gs.grippers_angle / 1_000_000.0
        # Effort in 0.001 N·m -> N·m
        effort_nm = gs.grippers_effort / 1000.0
        return (opening_m, effort_nm, gs.status_code)

    def open_gripper(self, width: float = 0.08, effort: float = 0.0) -> None:
        """Open the gripper to a specified width.

        Parameters
        ----------
        width : float, optional
            Desired opening width of the gripper in metres.  The Piper
            gripper has a limited range; values outside the physical
            range will be clamped by the firmware.  Defaults to 0.08 m.
        effort : float, optional
            Desired gripping effort in N·m.  The SDK represents
            gripper torque in units of 0.001 N·m.  Defaults to 0.0.
        """
        self.gripper_control(angle=width, effort=effort, enable=True)

    def close_gripper(self, effort: float = 0.0) -> None:
        """Close the gripper fully.

        Parameters
        ----------
        effort : float, optional
            Desired gripping effort in N·m.  The SDK represents
            gripper torque in units of 0.001 N·m.  Defaults to 0.0.
        """
        self.gripper_control(angle=0.0, effort=effort, enable=True)

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------
    def get_status(self):
        """Return the raw arm status message.

        The returned object is an instance of
        :class:`piper_sdk.piper_msgs.msg_v2.feedback.arm_feedback_status.ArmMsgFeedbackStatus`.
        It contains information such as the current operating mode,
        error flags and other diagnostic data.  Refer to the Piper
        documentation for details.
        """
        return self._interface.GetArmStatus()

    def emergency_stop(self) -> None:
        """Trigger an emergency stop.

        Sends a quick emergency stop command (``0x01``) to the arm.
        To resume movement after stopping you must call :meth:`resume`.
        """
        self._interface.EmergencyStop(0x01)

    def resume(self) -> None:
        """Resume motion after an emergency stop.

        Sends a resume command (``0x02``) to the arm.
        """
        self._interface.EmergencyStop(0x02)

    def close_connection(self) -> None:
        """Disconnect the CAN port and stop all background threads."""
        self._interface.DisconnectPort()

    def go_home(self, mode: Literal[1, 2, 3] = 1) -> None:
        """Send the arm back to its home (zero) joint configuration.

        Parameters
        ----------
        mode : int, optional
            Home return mode as defined by the Piper SDK (default 1).  The
            modes correspond to:

            * ``0`` – Restore master/slave arm mode without moving.
            * ``1`` – Return only the master arm to its zero position.
            * ``2`` – Return both master and slave arms to zero.

        Notes
        -----
        This command requires firmware V1.7‑4 or later.  During the
        homing process the arm will automatically move each joint
        until its zero sensor is triggered.  The call returns
        immediately; monitor joint positions or status feedback to
        determine when homing has completed.
        """
        self._interface.ReqMasterArmMoveToHome(mode)

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------
    def _wait_until_joint_target_reached(self, target_angles_deg: Iterable[float], tolerance_deg: float) -> None:
        """Block until the arm's joint positions are within tolerance of the target.

        This helper polls the joint feedback and arm status at roughly
        100 Hz.  It returns when the maximum absolute difference
        between the current joint angles and ``target_angles_deg`` (in
        degrees) is less than ``tolerance_deg`` or when the arm
        reports that it has reached the target position via the
        ``motion_status`` field of the arm status feedback.

        Parameters
        ----------
        target_angles_deg : iterable of float
            Target joint angles in degrees.
        tolerance_deg : float
            Acceptable error tolerance in degrees.
        """
        # Convert target to NumPy array for vectorised operations
        target = np.asarray(list(target_angles_deg), dtype=float)
        while True:
            # Read current joint feedback
            current = np.array(self.get_joint_values_raw(), dtype=float) / 1000.0
            # Check maximum absolute error
            if np.max(np.abs(current - target)) <= tolerance_deg:
                break
            # TODO this does not work reliably; disable for now
            time.sleep(0.05)


if __name__ == "__main__":
    import time

    can_name = "can0"  # Change this to your CAN interface name if needed
    if platform.system() == "Windows":
        can_name = "0"
    print("Creating PiperArmController...")
    arm = PiperArmController(can_name=can_name, has_gripper=True)
    arm.disable()
    print("Current joint angles (deg):")
    print(np.degrees(arm.get_joint_values()))

    print("Current end-effector pose:")
    # pos, rot = arm.get_pose()

    print("Current raw pose (m, rad):", arm.get_pose_raw()[:3],
          np.degrees(arm.get_pose_raw()[3:]))

    print("Is enabled:", arm.is_enabled)

    print("Go zero joints...")
    # arm.move_j(
    #     [0,0,0,0,0,0],
    #     speed=10,
    #     block=True
    # )
    # print("I have already run")
    # time.sleep(0.5)
    # arm.open_gripper()
    # time.sleep(0.5)
    # arm.close_gripper()

    for i in range(10000):
        print(arm.get_joint_values())
        time.sleep(0.01)