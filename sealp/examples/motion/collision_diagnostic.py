"""
Piper Collision Detection Diagnostic
=======================================

Tests and visualizes the Piper robot's collision detection in
three scenarios:

1. **Self-collision check** — robot in normal vs self-colliding poses
2. **Obstacle collision** — robot arm vs environment obstacle
3. **Hold + collision** — robot holding an object, checking held
   object vs obstacle collision

Compares behavior with the Cobotta robot as reference.

Usage::

    python -m sealp.examples.motion.collision_diagnostic
"""

import os
import sys
import numpy as np
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from wrs import wd, rm, mgm, mcm


def test_piper_collision():
    """Test Piper collision detection."""
    import wrs.robot_sim.robots.piper.piper_single_arm as psa

    base = wd.World(cam_pos=[1.5, 1.0, 1.0], lookat_pos=[0.2, 0, 0.2])
    mgm.gen_frame().attach_to(base)

    robot = psa.PiperSglArm(enable_cc=True)

    # ------------------------------------------------------------------
    # Test 1: Self-collision at home config (should NOT collide)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("TEST 1: Self-collision at home configuration")
    print("=" * 60)
    robot.goto_given_conf(np.zeros(6))
    is_collided = robot.is_collided()
    print(f"  Home conf collided: {is_collided}")
    assert not is_collided, "Home configuration should NOT be collided!"
    robot.gen_meshmodel(alpha=0.3, toggle_cdprim=True).attach_to(base)

    # ------------------------------------------------------------------
    # Test 2: Obstacle collision
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 2: Robot vs obstacle collision")
    print("=" * 60)

    # Place a box in the workspace
    box = mcm.gen_box(
        xyz_lengths=np.array([0.1, 0.1, 0.1]),
        rgb=np.array([1, 0.3, 0.3]), alpha=0.5)
    box.pos = np.array([0.25, 0.0, 0.15])
    box.attach_to(base)

    # Move robot near the box
    test_conf = np.array([0, 0.3, 0.5, 0, 0.3, 0])
    robot.goto_given_conf(test_conf)
    is_collided_no_obs = robot.is_collided()
    is_collided_with_obs = robot.is_collided(obstacle_list=[box])
    print(f"  Conf collided (no obstacle): {is_collided_no_obs}")
    print(f"  Conf collided (with box):    {is_collided_with_obs}")
    robot.gen_meshmodel(alpha=0.5, toggle_cdprim=True).attach_to(base)

    # ------------------------------------------------------------------
    # Test 3: Hold object + collision check
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 3: Hold object + collision check")
    print("=" * 60)

    # Create an object to hold
    held_obj = mcm.gen_box(
        xyz_lengths=np.array([0.04, 0.03, 0.06]),
        rgb=np.array([0.3, 0.8, 0.3]), alpha=0.8)

    # Move to a known reachable pose
    tgt_pos = np.array([0.2, 0.1, 0.20])
    tgt_rotmat = rm.rotmat_from_euler(0, math.pi / 2, 0)
    jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)

    if jnt_values is not None:
        robot.goto_given_conf(jnt_values)
        # Place the object at the TCP
        tcp_pos, tcp_rotmat = robot.gl_tcp_pos, robot.gl_tcp_rotmat
        held_obj.pos = tcp_pos
        held_obj.rotmat = tcp_rotmat

        print(f"  TCP pos: {tcp_pos}")
        print(f"  Holding object at TCP...")

        # Hold the object
        robot.hold(obj_cmodel=held_obj, jaw_width=0.02)

        # Check collision with nearby obstacle
        nearby_obs = mcm.gen_box(
            xyz_lengths=np.array([0.05, 0.05, 0.05]),
            rgb=np.array([0.8, 0.8, 0.3]), alpha=0.5)
        nearby_obs.pos = tcp_pos + np.array([0.0, 0.0, 0.08])
        nearby_obs.attach_to(base)

        is_collided_held = robot.is_collided(obstacle_list=[nearby_obs])
        print(f"  Held object collided with nearby box: {is_collided_held}")

        # Also check mesh collision of end effector
        is_mesh_collided = robot.end_effector.is_mesh_collided(
            cmodel_list=[nearby_obs])
        print(f"  EE mesh collided with nearby box: {is_mesh_collided}")

        robot.gen_meshmodel(alpha=0.7, toggle_cdprim=True).attach_to(base)

        # Release
        robot.release(obj_cmodel=held_obj, jaw_width=0.05)
    else:
        print("  IK failed for test pose, skipping hold test.")

    # ------------------------------------------------------------------
    # Test 4: Show cdprim to verify visual alignment
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 4: Visual cdprim inspection")
    print("=" * 60)
    robot.goto_given_conf(np.zeros(6))
    robot.show_cdprim()
    print("  Showing collision primitives (green outlines).")
    print("  Verify they align with the robot mesh.")

    print("\n" + "=" * 60)
    print("All tests completed. Press ESC to close.")
    print("=" * 60)
    base.run()


def test_cobotta_reference():
    """Test Cobotta collision as reference (known working)."""
    import wrs.robot_sim.robots.cobotta.cobotta as cbt

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, 0.3])
    mgm.gen_frame().attach_to(base)

    robot = cbt.Cobotta(enable_cc=True)

    print("=" * 60)
    print("COBOTTA REFERENCE: Self-collision check")
    print("=" * 60)

    # Home config
    is_collided = robot.is_collided()
    print(f"  Home conf collided: {is_collided}")

    # Obstacle
    box = mcm.gen_box(
        xyz_lengths=np.array([0.1, 0.1, 0.1]),
        rgb=np.array([1, 1, 0]), alpha=0.3)
    box.pos = np.array([0.15, 0, 0.15])
    box.attach_to(base)

    is_collided_obs = robot.is_collided(obstacle_list=[box])
    print(f"  With obstacle: {is_collided_obs}")

    robot.gen_meshmodel(alpha=0.5, toggle_cdprim=False).attach_to(base)
    # robot.show_cdprim()

    print(f"\nPress ESC to close.")
    base.run()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "cobotta":
        test_cobotta_reference()
    else:
        test_piper_collision()
