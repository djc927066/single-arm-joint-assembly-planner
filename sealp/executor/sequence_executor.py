"""
Sequence Executor
==================

Execute an assembly sequence step-by-step, dispatching each step
to the appropriate motion primitive and dynamically updating the
obstacle list as parts are placed.

Supports both ``.asmdef`` + ``.tplan`` format and legacy
``AssemblySequence`` format.

Key pattern (from ``pick_and_place_chair.py``)::

    for step in assembly.get_execution_order():
        obj = parts[step.part_id]
        obstacle_this_round = [o for o in obs_list if o is not obj]
        mot = planner.gen_pick_and_place(
            obj_cmodel=obj,
            obstacle_list=obstacle_this_round, ...)
        current_conf = mot.jv_list[-1]
        placed = obj.copy(); placed.pos = goal_pos
        obs_list.append(placed)

Usage::

    from sealp.executor import SequenceExecutor

    executor = SequenceExecutor(
        robot=dual_robot,
        assembly_def=asm,
        task_plan=plan,
        obstacle_list=[ground],
    )
    result = executor.execute_all()
    for step_result in result.steps:
        print(f"Step {step_result.step_id}: {step_result.success}")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pickle
import wrs.modeling.collision_model as mcm
import wrs.basis.robot_math as rm

from sealp.assembly_sequence.primitives import Primitive
from sealp.assembly_sequence.asmdef import AssemblyDef, StepDef
from sealp.assembly_sequence.tplan import TaskPlan, StepParams
from sealp.primitives.base import PrimitiveResult
from .primitive_selector import PrimitiveSelector


# ══════════════════════════════════════════════════════════════
#  Result data classes
# ══════════════════════════════════════════════════════════════
@dataclass
class StepResult:
    """Result of planning a single assembly step.

    Attributes
    ----------
    step_id : int
        Assembly step index.
    part_id : str
        Part that was assembled.
    primitive : str
        Motion primitive type used (value from ``Primitive`` enum).
    success : bool
        Whether planning succeeded.
    mot_data : object or None
        Single-arm MotionData.
    mot_data_rgt : object or None
        Right-arm MotionData (dual-arm steps).
    mot_data_lft : object or None
        Left-arm MotionData (dual-arm steps).
    end_conf : np.ndarray or None
        Joint configuration after this step.
    end_conf_rgt : np.ndarray or None
        Right-arm joint conf after this step (dual-arm).
    end_conf_lft : np.ndarray or None
        Left-arm joint conf after this step (dual-arm).
    error_msg : str
        Error description if planning failed.
    """
    step_id: int = -1
    part_id: str = ""
    primitive: str = ""
    success: bool = False
    mot_data: object = None
    mot_data_rgt: object = None
    mot_data_lft: object = None
    end_conf: Optional[np.ndarray] = None
    end_conf_rgt: Optional[np.ndarray] = None
    end_conf_lft: Optional[np.ndarray] = None
    error_msg: str = ""

    @property
    def n_frames(self) -> int:
        """Total animation frames in this step."""
        total = 0
        if self.mot_data is not None:
            total += len(self.mot_data)
        if self.mot_data_rgt is not None:
            total += len(self.mot_data_rgt)
        if self.mot_data_lft is not None:
            total += len(self.mot_data_lft)
        return total


@dataclass
class ExecutionResult:
    """Result of executing an entire assembly sequence.

    Attributes
    ----------
    steps : list of StepResult
        Per-step results.
    success : bool
        True if all steps succeeded.
    total_frames : int
        Total animation frames across all steps.
    """
    steps: List[StepResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return all(s.success for s in self.steps)

    @property
    def total_frames(self) -> int:
        return sum(s.n_frames for s in self.steps)

    @property
    def n_succeeded(self) -> int:
        return sum(1 for s in self.steps if s.success)

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.steps if not s.success)

    def summary(self) -> str:
        lines = [
            f"Execution Result: {self.n_succeeded}/{len(self.steps)} steps OK",
            f"Total frames: {self.total_frames}",
        ]
        for s in self.steps:
            status = "✅" if s.success else "❌"
            frames = f"{s.n_frames} frames" if s.success else s.error_msg
            lines.append(
                f"  {status} Step {s.step_id} ({s.part_id}): "
                f"{s.primitive} — {frames}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  Sequence Executor
# ══════════════════════════════════════════════════════════════
class SequenceExecutor:
    """Execute an assembly sequence step-by-step.

    For each step in topological order:
    1. Load/create the part's collision model.
    2. Plan or load grasps for the part.
    3. Dispatch to the motion primitive (transport / dual_transport).
    4. On success: move placed part to obstacle list.
    5. Track joint configurations across steps.

    Parameters
    ----------
    robot : DualPiperNoBody or SglArmRobotInterface
        Robot to use.  If a dual-arm robot, ``robot.rgt_arm`` is used
        for single-arm steps and both arms for dual-arm steps.
        If a single-arm robot, dual-arm steps will fail.
    assembly_def : AssemblyDef
        Assembly definition (parts + steps + models).
    task_plan : TaskPlan or None
        Task plan with staging poses and step params.
        If ``None``, basic defaults are used.
    obstacle_list : list or None
        Initial obstacles (ground plane, fixtures, etc.).
    assembled_cd_ex_radius : float or None
        If set, each successfully placed part is appended to the working
        obstacle list with this CD primitive expansion (meters).  Larger
        values make later steps stay farther from assembled geometry; ``None``
        keeps the model default thickness (~2 mm).
    grasp_cache : dict or None
        Pre-loaded grasps: ``{model_alias: GraspCollection}``.
        If ``None``, grasps are planned on-the-fly.
    grasp_paths : dict or None
        Optional paths to pickled grasp files per model alias.
    """

    def __init__(self,
                 robot,
                 assembly_def: AssemblyDef,
                 task_plan: Optional[TaskPlan] = None,
                 obstacle_list: Optional[List] = None,
                 grasp_cache: Optional[Dict] = None,
                 grasp_paths: Optional[Dict[str, str]] = None,
                 assembled_cd_ex_radius: Optional[float] = None):
        self.assembly_def = assembly_def
        self.task_plan = task_plan
        self._initial_obstacles = list(obstacle_list or [])
        self._assembled_cd_ex_radius = assembled_cd_ex_radius
        self._grasp_cache = dict(grasp_cache or {})
        self._grasp_paths = dict(grasp_paths or {})

        # ── Detect robot type ────────────────────────────────
        self._is_dual = hasattr(robot, 'rgt_arm') and hasattr(robot, 'lft_arm')
        if self._is_dual:
            self._robot = robot
            self._robot_rgt = robot.rgt_arm
            self._robot_lft = robot.lft_arm
        else:
            self._robot = robot
            self._robot_rgt = robot
            self._robot_lft = None

        # ── Primitive selector ───────────────────────────────
        self._selector = PrimitiveSelector(
            robot_rgt=self._robot_rgt,
            robot_lft=self._robot_lft,
        )

        # ── Part collision models (loaded lazily) ────────────
        self._part_cmodels: Dict[str, mcm.CollisionModel] = {}

    # ── Public API ───────────────────────────────────────────
    def execute_all(self,
                    stop_on_failure: bool = True) -> ExecutionResult:
        """Plan all steps in topological order.

        Parameters
        ----------
        stop_on_failure : bool
            If True, stop at the first failed step.
            If False, continue and mark failed steps.

        Returns
        -------
        ExecutionResult
        """
        result = ExecutionResult()
        steps = self.assembly_def.get_execution_order()

        # Working obstacle list (grows as parts are placed)
        obs_list = list(self._initial_obstacles)

        # Current joint config (carried forward)
        current_conf_rgt = self._robot_rgt.get_jnt_values()
        current_conf_lft = (
            self._robot_lft.get_jnt_values()
            if self._robot_lft is not None else None
        )

        # Compute world assembly poses
        fixture_pos = np.zeros(3)
        fixture_rotmat = np.eye(3)
        if self.task_plan is not None:
            fixture_pos = self.task_plan.fixture_pos
            fixture_rotmat = self.task_plan.fixture_rotmat
        world_poses = self.assembly_def.compute_world_poses(
            fixture_pos=fixture_pos, fixture_rotmat=fixture_rotmat)

        print("=" * 60)
        print(f"Sequence Executor: {self.assembly_def.name}")
        print(f"  Steps: {len(steps)}  |  Obstacles: {len(obs_list)}")
        print("=" * 60)

        for i, step in enumerate(steps):
            print(f"\n── Step {step.step_id}: assemble {step.part_id!r} "
                  f"({i + 1}/{len(steps)}) ──")

            step_result = self._execute_step(
                step=step,
                world_poses=world_poses,
                obs_list=obs_list,
                current_conf_rgt=current_conf_rgt,
                current_conf_lft=current_conf_lft,
            )
            result.steps.append(step_result)

            if step_result.success:
                # Update joint configs
                if step_result.end_conf is not None:
                    current_conf_rgt = step_result.end_conf
                if step_result.end_conf_rgt is not None:
                    current_conf_rgt = step_result.end_conf_rgt
                if step_result.end_conf_lft is not None:
                    current_conf_lft = step_result.end_conf_lft

                # Add placed part to obstacles (optional inflated CD for later steps)
                goal_pos, goal_rotmat = world_poses[step.part_id]
                placed = self._get_part_cmodel(step.part_id).copy()
                placed.pos = goal_pos
                placed.rotmat = goal_rotmat
                if self._assembled_cd_ex_radius is not None:
                    try:
                        placed.change_cdprim_type(
                            cdprim_type=placed.cdprim_type,
                            ex_radius=self._assembled_cd_ex_radius,
                        )
                    except Exception as _err:
                        # 不再静默吞异常 —— 让上层看到 cdprim 重建失败的真实
                        # 原因。即使失败，placed 仍以原始 cdprim 加入 obs_list，
                        # 不阻塞执行，但避障会比预期严格不足。
                        print(f"  [WARN] {step.part_id} change_cdprim_type "
                              f"(ex_radius={self._assembled_cd_ex_radius}) "
                              f"failed: {_err!r}")
                placed._sealp_role = "assembled_at_goal"
                placed._sealp_part_id = step.part_id
                obs_list.append(placed)
                print(f"  ✅ Success ({step_result.n_frames} frames)")
            else:
                print(f"  ❌ FAILED: {step_result.error_msg}")
                if stop_on_failure:
                    print("  Stopping execution (stop_on_failure=True).")
                    break

        print("\n" + "=" * 60)
        print(result.summary())
        print("=" * 60)
        return result

    # ── Internal ─────────────────────────────────────────────
    def _execute_step(self,
                      step: StepDef,
                      world_poses: Dict,
                      obs_list: List,
                      current_conf_rgt: np.ndarray,
                      current_conf_lft: Optional[np.ndarray],
                      ) -> StepResult:
        """Plan a single assembly step."""
        part_id = step.part_id

        # ── Determine primitive type ─────────────────────────
        primitive_type = Primitive.SINGLE_ARM_TRANSPORT  # default
        step_params = None
        if self.task_plan is not None:
            step_params = self.task_plan.get_step_params(step.step_id)
            if step_params is not None:
                primitive_type = step_params.primitive

        # ── Get object collision model ────────────────────────
        obj = self._get_part_cmodel(part_id)

        # ── Plan grasps at ORIGIN first ──────────────────────
        # WRS's sample_surface() transforms contact points by
        # obj.homomat (pos + rotmat).  Grasps must be planned
        # with the object at origin so ac_pos is in local frame.
        # gen_pick_and_place then correctly computes world TCP as:
        #   tcp_pos = obj.rotmat @ ac_pos + obj.pos
        part_def = self.assembly_def.get_part(part_id)
        obj.pos = np.zeros(3)
        obj.rotmat = np.eye(3)
        grasp_collection = self._get_grasps(part_def.model, obj)

        # ── Now set staging (pick) pose ──────────────────────
        staging_pos, staging_rotmat = self._get_staging_pose(part_id)
        obj.pos = staging_pos
        obj.rotmat = staging_rotmat

        # ── Get goal (assembly) pose ─────────────────────────
        if part_id not in world_poses:
            return StepResult(
                step_id=step.step_id, part_id=part_id,
                primitive=primitive_type.value,
                error_msg=f"No world pose computed for {part_id!r}.")

        goal_pos, goal_rotmat = world_poses[part_id]
        goal_pose_list = [(goal_pos, goal_rotmat)]
        if grasp_collection is None or len(grasp_collection) == 0:
            return StepResult(
                step_id=step.step_id, part_id=part_id,
                primitive=primitive_type.value,
                error_msg=f"No grasps available for {part_id!r} "
                          f"(model={part_def.model!r}).")

        # ── Prepare obstacle list (exclude this part) ────────
        obstacle_this_round = [o for o in obs_list]

        # ── Approach / depart params ─────────────────────────
        approach_dist = 0.05
        depart_dist = 0.05
        if step_params is not None:
            approach_dist = step_params.approach_distance
            depart_dist = step_params.depart_distance

        # ── Select and run primitive ─────────────────────────
        try:
            prim = self._selector.select(primitive_type)
        except (ValueError, NotImplementedError) as e:
            return StepResult(
                step_id=step.step_id, part_id=part_id,
                primitive=primitive_type.value,
                error_msg=str(e))

        # Build kwargs
        plan_kwargs = dict(
            obj_cmodel=obj,
            grasp_collection=grasp_collection,
            goal_pose_list=goal_pose_list,
            obstacle_list=obstacle_this_round,
            approach_distance=approach_dist,
            depart_distance=depart_dist,
            use_rrt=True,
            part_id=part_id,
        )

        if primitive_type == Primitive.SINGLE_ARM_TRANSPORT:
            plan_kwargs["start_jnt_values"] = None
            plan_kwargs["end_jnt_values"] = current_conf_rgt
        elif primitive_type == Primitive.DUAL_ARM_COOPERATIVE:
            plan_kwargs["start_jnt_values_rgt"] = None
            plan_kwargs["end_jnt_values_rgt"] = current_conf_rgt
            if current_conf_lft is not None:
                plan_kwargs["start_jnt_values_lft"] = None
                plan_kwargs["end_jnt_values_lft"] = current_conf_lft

        prim_result: PrimitiveResult = prim.plan(**plan_kwargs)

        # ── Build StepResult ─────────────────────────────────
        return StepResult(
            step_id=step.step_id,
            part_id=part_id,
            primitive=primitive_type.value,
            success=prim_result.success,
            mot_data=prim_result.mot_data,
            mot_data_rgt=prim_result.mot_data_rgt,
            mot_data_lft=prim_result.mot_data_lft,
            end_conf=prim_result.end_jnt_values,
            end_conf_rgt=prim_result.end_jnt_values_rgt,
            end_conf_lft=prim_result.end_jnt_values_lft,
            error_msg=prim_result.error_msg,
        )

    # ── Part collision model management ──────────────────────
    def _get_part_cmodel(self, part_id: str) -> mcm.CollisionModel:
        """Load or return a cached collision model for a part."""
        if part_id in self._part_cmodels:
            return self._part_cmodels[part_id]

        part_def = self.assembly_def.get_part(part_id)
        model_path = self.assembly_def.model_path(part_id)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model file not found for part {part_id!r}: {model_path}")

        cmodel = mcm.CollisionModel(initor=model_path)
        self._part_cmodels[part_id] = cmodel
        return cmodel

    # ── Staging pose ─────────────────────────────────────────
    def _get_staging_pose(
            self, part_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the staging (pick) pose for a part."""
        if self.task_plan is not None:
            staging = self.task_plan.get_staging(part_id)
            if staging is not None:
                return staging.pos.copy(), staging.rotmat.copy()

        # Fallback: use the assembly_def's first step's relative pos
        # with some offset (not ideal, but a reasonable default)
        print(f"  WARNING: No staging pose for {part_id!r}, "
              f"using origin.")
        return np.zeros(3), np.eye(3)

    # ── Grasp management ─────────────────────────────────────
    # def _get_grasps(self, model_alias: str, obj_cmodel):
    #     """Get or plan grasps for a model alias."""
    #     if model_alias in self._grasp_cache:
    #         return self._grasp_cache[model_alias]
    #
    #     # Plan grasps on-the-fly
    #     print(f"  Planning grasps for model {model_alias!r}...")
    #     try:
    #         from sealp.examples.grasp.planning import plan_grasps
    #         grasp_collection, _ = plan_grasps(
    #             obj_cmodel, max_samples=50)
    #         self._grasp_cache[model_alias] = grasp_collection
    #         print(f"  Planned {len(grasp_collection)} grasps.")
    #         return grasp_collection
    #     except Exception as e:
    #         print(f"  ERROR planning grasps: {e}")
    #         return None

    # ── Grasp management ─────────────────────────────────────
    def _get_grasps(self, model_alias: str, obj_cmodel):
        """Get or plan grasps for a model alias."""
        if model_alias in self._grasp_cache:
            return self._grasp_cache[model_alias]
        if model_alias in self._grasp_paths:
            file_path = self._grasp_paths[model_alias]
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                print(f"  [底层拦截] 正在从自定义路径加载 {model_alias!r} 的抓取数据...")
                print(f"  路径: {file_path}")
                with open(file_path, 'rb') as f:
                    grasps = pickle.load(f)
                    self._grasp_cache[model_alias] = grasps
                    print(f"成功加载 {len(grasps)} 个姿态！")
                    return grasps
            else:
                print(f"  [底层警告] 指定的路径不存在或文件为空: {file_path}")
                print(f"  将退回使用实时盲算！")
        print(f"  Planning grasps for model {model_alias!r}...")
        try:
            from sealp.examples.grasp.planning import plan_grasps
            grasp_collection, _ = plan_grasps(
                obj_cmodel, max_samples=50)
            self._grasp_cache[model_alias] = grasp_collection
            print(f"  Planned {len(grasp_collection)} grasps.")
            return grasp_collection
        except Exception as e:
            print(f"  ERROR planning grasps: {e}")
            return None

    # ── Convenience ──────────────────────────────────────────
    def set_grasp_cache(self, model_alias: str, grasp_collection):
        """Pre-load grasps for a model alias."""
        self._grasp_cache[model_alias] = grasp_collection
