"""
Sequence Execution Demo — YuanChair Assembly
=============================================

Demonstrates the full assembly execution pipeline:

1. Load the YuanChair ``.asmdef`` assembly definition.
2. Create a ``TaskPlan`` with staging positions.
3. Create a ``SequenceExecutor`` with a Piper arm.
4. Execute all assembly steps in topological order.
5. Animate the results in Panda3D (press **Space** to step).

This is the reference example for Phase 2: Sequential Manipulation.

Usage::

    python -m sealp.examples.motion.sequence_execution

Prerequisites:
    - Run ``python -m sealp.assembly_sequence.gen_yuanchair_asmdef``
      to generate the ``.asmdef`` file (if not already present).
"""

import os
import sys
import numpy as np

from wrs import wd, rm, mgm, mcm
from direct.task.TaskManagerGlobal import taskMgr

# Ensure PYTHONPATH includes the WRS root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from sealp.assembly_sequence import AssemblyDef, TaskPlan, StepParams
from sealp.executor import SequenceExecutor


# ══════════════════════════════════════════════════════════════
#  Animation
# ══════════════════════════════════════════════════════════════
def animate_sequence(base, execution_result, interval=0.01):
    """Animate all successful steps sequentially.

    Each step's motion plays in order.  Press **Space** to advance
    frames.

    Parameters
    ----------
    base : wd.World
        Panda3D world.
    execution_result : ExecutionResult
        Result from ``SequenceExecutor.execute_all()``.
    interval : float
        Delay between frames (seconds).
    """
    # Collect all motion data into a flat list of (step_id, mot_data) pairs
    all_motions = []
    for sr in execution_result.steps:
        if not sr.success:
            continue
        if sr.mot_data is not None:
            all_motions.append((sr.step_id, sr.part_id, sr.mot_data))
        # For dual-arm steps: animate right arm
        # (left arm would need dual animation — future enhancement)
        elif sr.mot_data_rgt is not None:
            all_motions.append(
                (sr.step_id, sr.part_id, sr.mot_data_rgt))

    if not all_motions:
        print("No motions to animate.")
        return

    class _State:
        def __init__(self):
            self.step_idx = 0
            self.frame_idx = 0
            self.total_steps = len(all_motions)

    state = _State()

    def _update(st, task):
        if st.step_idx >= st.total_steps:
            # Loop back to start
            st.step_idx = 0
            st.frame_idx = 0

        step_id, part_id, mot = all_motions[st.step_idx]

        # Detach previous frame
        if st.frame_idx > 0:
            mot.mesh_list[st.frame_idx - 1].detach()
        elif st.step_idx > 0:
            # Detach last frame of previous step
            _, _, prev_mot = all_motions[st.step_idx - 1]
            if len(prev_mot.mesh_list) > 0:
                prev_mot.mesh_list[-1].detach()

        if st.frame_idx >= len(mot.mesh_list):
            # Detach all meshes from this step
            for m in mot.mesh_list:
                m.detach()
            st.step_idx += 1
            st.frame_idx = 0
            return task.again

        # Show current frame
        mesh = mot.mesh_list[st.frame_idx]
        mesh.attach_to(base)

        if base.inputmgr.keymap['space']:
            st.frame_idx += 1

        return task.again

    taskMgr.doMethodLater(interval, _update, "sequence_animate",
                          extraArgs=[state], appendTask=True)


# ══════════════════════════════════════════════════════════════
#  Demo
# ══════════════════════════════════════════════════════════════
def main():
    """Execute the YuanChair assembly step-by-step."""
    import wrs.robot_sim.robots.piper.piper_single_arm as psa

    # ------------------------------------------------------------------
    # 1. Scene setup
    # ------------------------------------------------------------------
    base = wd.World(cam_pos=[1.2, 0.7, 1.0], lookat_pos=[0.3, 0, 0.1])
    mgm.gen_frame().attach_to(base)

    # Ground plane
    ground = mcm.gen_box(
        xyz_lengths=rm.vec(2, 2, 0.01),
        rgb=rm.vec(0.75, 0.75, 0.75), alpha=1)
    ground.pos = np.array([0.3, 0, -0.01])
    ground.attach_to(base)

    # ------------------------------------------------------------------
    # 2. Load / generate the assembly definition
    # ------------------------------------------------------------------
    asmdef_dir = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "assembly_sequence", "_demo_output")
    asmdef_path = os.path.abspath(
        os.path.join(asmdef_dir, "yuanchair.asmdef"))

    if not os.path.isfile(asmdef_path):
        print(f"Assembly definition not found at: {asmdef_path}")
        print("Generating it now...")
        from sealp.assembly_sequence.gen_yuanchair_asmdef import main as gen_asm
        gen_asm()

    if not os.path.isfile(asmdef_path):
        print(f"ERROR: Could not find or generate {asmdef_path}")
        return

    asm = AssemblyDef.load(asmdef_path)
    print(f"Loaded: {asm.name} ({asm.n_parts} parts, {asm.n_steps} steps)")

    # ------------------------------------------------------------------
    # 3. Create task plan with staging positions
    # ------------------------------------------------------------------
    plan = TaskPlan(
        assembly_file=asmdef_path,
        name="YuanChair Demo Execution",
        description="Single-arm sequential assembly of the YuanChair.",
    )
    plan.set_assembly(asm)

    # Fixture at origin
    plan.fixture_pos = np.array([0.0, 0.0, 0.0])
    plan.fixture_rotmat = np.eye(3)

    # Staging positions (from pick_and_place_chair.py)
    plan.set_staging("seat",   pos=np.array([0.30,  0.00, 0.00]))
    plan.set_staging("leg_fl", pos=np.array([0.15,  0.25, 0.00]))
    plan.set_staging("leg_fr", pos=np.array([0.2, -0.24, 0.00]))
    plan.set_staging("leg_bl", pos=np.array([0.25,  0.24, 0.00]))
    plan.set_staging("leg_br", pos=np.array([0.25, -0.24, 0.00]))

    # Step params (all single-arm transport for this demo)
    for i in range(asm.n_steps):
        plan.set_step_params(StepParams(
            step_id=i,
            primitive="single_arm_transport",
            approach_distance=0.06,
            depart_distance=0.02,
        ))

    # ------------------------------------------------------------------
    # 4. Visualize assembly goal poses (translucent ghosts)
    # ------------------------------------------------------------------
    world_poses = asm.compute_world_poses(
        fixture_pos=plan.fixture_pos,
        fixture_rotmat=plan.fixture_rotmat,
    )
    for pid in asm.part_ids:
        if pid not in world_poses:
            continue
        goal_pos, goal_rotmat = world_poses[pid]
        model_path = asm.model_path(pid)
        if os.path.isfile(model_path):
            ghost = mcm.CollisionModel(initor=model_path)
            ghost.pos = goal_pos
            ghost.rotmat = goal_rotmat
            ghost.alpha = 0.15
            ghost.attach_to(base)

    # ------------------------------------------------------------------
    # 4b. Visualize parts at staging (initial/pick) positions
    # ------------------------------------------------------------------
    staging_colors = {
        "seat":   np.array([0.9, 0.6, 0.3, 0.7]),
        "leg_fl": np.array([0.3, 0.7, 0.3, 0.7]),
        "leg_fr": np.array([0.3, 0.3, 0.8, 0.7]),
        "leg_bl": np.array([0.8, 0.3, 0.3, 0.7]),
        "leg_br": np.array([0.7, 0.3, 0.7, 0.7]),
    }
    print("\nStaging positions:")
    for pid in asm.part_ids:
        staging = plan.get_staging(pid)
        if staging is None:
            continue
        model_path = asm.model_path(pid)
        if os.path.isfile(model_path):
            staged = mcm.CollisionModel(initor=model_path)
            staged.pos = staging.pos
            staged.rotmat = staging.rotmat
            staged.rgba = staging_colors.get(
                pid, np.array([0.5, 0.5, 0.5, 0.7]))
            staged.attach_to(base)
            # Show a small frame at staging position
            mgm.gen_frame(pos=staging.pos, ax_length=0.03).attach_to(base)
            print(f"  {pid}: pos={staging.pos.tolist()}")

    # ------------------------------------------------------------------
    # 5. Robot
    # ------------------------------------------------------------------
    robot = psa.PiperSglArm(enable_cc=True)
    robot.gen_meshmodel(alpha=0.2).attach_to(base)

    initial_obstacles = [ground]

    for pid in asm.part_ids:
        staging = plan.get_staging(pid)
        if staging is None:
            continue
        model_path = asm.model_path(pid)
        if os.path.isfile(model_path):
            staged_obs = mcm.CollisionModel(initor=model_path)
            staged_obs.pos = staging.pos
            staged_obs.rotmat = staging.rotmat
            initial_obstacles.append(staged_obs)
    # import pickle
    #
    # # 1. 填入你想要使用的、由 filtering.py 过滤好的新 pickle 文件路径
    # leg_pickle_path = r"D:\Project\wrs-sealp\sealp\examples\grasp\_output\demo_yuanchair-part2_filter_grasps.pickle"
    # seat_pickle_path = r"D:\Project\wrs-sealp\sealp\examples\grasp\_output\demo_yuanchair-part1_filter_grasps.pickle"
    # custom_grasp_cache = {}
    # if os.path.exists(leg_pickle_path):
    #     with open(leg_pickle_path, 'rb') as f:
    #         custom_grasp_cache['leg_model'] = pickle.load(f)
    #         print(f"[INFO] 成功加载椅腿抓取数据: {len(custom_grasp_cache['leg_model'])} 个")
    # else:
    #     print(f"[WARNING] 找不到椅腿抓取文件: {leg_pickle_path}")
    # if os.path.exists(seat_pickle_path):
    #     with open(seat_pickle_path, 'rb') as f:
    #         custom_grasp_cache['seat_model'] = pickle.load(f)
    #         print(f"[INFO] 成功加载座椅抓取数据: {len(custom_grasp_cache['seat_model'])} 个")

    # ------------------------------------------------------------------
    # 6. Execute
    # ------------------------------------------------------------------
    executor = SequenceExecutor(
        robot=robot,
        assembly_def=asm,
        task_plan=plan,
        obstacle_list=[ground],
        grasp_paths={
            'leg_model': r"D:\Project\wrs-sealp\sealp\examples\grasp\_output\demo_yuanchair-part2_filter_grasps.pickle",
            'seat_model': r"D:\Project\wrs-sealp\sealp\examples\grasp\_output\demo_yuanchair-part1_grasps.pickle"
        }
    )

    print("\nExecuting assembly sequence...")
    result = executor.execute_all(stop_on_failure=False)

    # ------------------------------------------------------------------
    # 7. Animate
    # ------------------------------------------------------------------
    if result.n_succeeded > 0:
        print(f"\nPress SPACE to step through the animation "
              f"({result.total_frames} frames).")
        animate_sequence(base, result)
    else:
        print("\nNo successful steps to animate.")

    base.run()


if __name__ == "__main__":
    main()
