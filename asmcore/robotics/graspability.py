"""在**当前已装配障碍环境**下统计某零件的合格抓取数，即论文二的 G 质量。

逐字迁移自 WRS3 `wrs/assembly/robotics/graspability.py`。判据、容差、缓存键、
结果字段一律未改；唯一的改动是夹爪与抓取记录改走 `grasp_model` 的端口，
因此本模块不再 import 任何 `wrs.*`。

它统计的是力闭合成立、且夹爪在目标位姿下不与零件自身或已装零件碰撞的抓取数。
它**不**声称 IK、接近路径或搬运路径可行——那属于论文一的 L1/L3 层，由
`adapters/` 的 `GraspFeasibility` 与 `MotionWitness` 负责。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..geometry.proximity import MeshProximity
from ..mechanics.force_closure import check_force_closure
from ..model import Assembly, AssemblyState, digest, freeze
from .grasp_geometry import finger_force_contacts, generate_grasps, grasp_contacts
from .grasp_model import GraspRecord, GripperPort, gripper_signature


@dataclass(frozen=True)
class GraspabilityConfig:
    max_grasps: int = 12
    seed: int = 7
    friction: float = 0.5
    cone_sides: int = 16
    contact_roundoff_m: float = 2e-6
    pad_seating_tol_m: float | None = None
    pad_patch_tol_m: float | None = None
    collision_budget: int = 150000

    def __post_init__(self) -> None:
        for k in ("max_grasps", "collision_budget", "cone_sides"):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError(k)
        if self.cone_sides < 4 or type(self.seed) is not int:
            raise ValueError("Invalid cone/seed")
        if not np.isfinite(self.friction) or self.friction < 0:
            raise ValueError("Invalid friction")
        if not np.isfinite(self.contact_roundoff_m) or self.contact_roundoff_m <= 0:
            raise ValueError("Invalid roundoff")
        for key in ("pad_seating_tol_m", "pad_patch_tol_m"):
            value = getattr(self, key)
            if value is not None and (
                not np.isfinite(value) or value < self.contact_roundoff_m
            ):
                raise ValueError(key)


@dataclass(frozen=True)
class GraspabilityResult:
    status: str
    count: int | None
    accepted_ids: tuple[int, ...]
    catalogue_size: int
    input_digest: str
    diagnostics: dict

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted_ids", tuple(self.accepted_ids))
        object.__setattr__(self, "diagnostics", freeze(self.diagnostics))


class GraspabilityAnalyzer:
    """自有夹爪副本、不可变的抓取目录快照、按状态绑定的缓存。

    `grasps` 可选，给某些零件直接提供候选抓取。没有确认垫片贴合面的曲面接触
    保持 unresolved；仅有对踵点不构成 6D 力闭合的证明。

    满足 `asmcore.planning.ports.GraspabilityPort`，可直接交给
    `plan_quality_sequence`。
    """

    def __init__(
        self,
        assembly: Assembly,
        gripper: GripperPort,
        *,
        config: GraspabilityConfig | None = None,
        grasps: Mapping[str, Iterable[GraspRecord]] | None = None,
    ) -> None:
        self.assembly = assembly
        self.config = config or GraspabilityConfig()
        self._gripper = gripper.clone()
        self._parts = {p.part_id: p for p in assembly.parts}
        self._provided = {
            k: tuple(freeze(g.to_dict()) for g in v) for k, v in (grasps or {}).items()
        }
        if not set(self._provided).issubset(self._parts):
            raise ValueError("Unknown grasp part")
        self._catalogues = {}
        self._environment = {}
        self._results = {}
        self._backend = MeshProximity(numerical_tol_m=self.config.contact_roundoff_m)
        self._pad_seating_tol_m = float(
            self.config.pad_seating_tol_m
            if self.config.pad_seating_tol_m is not None
            else getattr(self._gripper, "pad_seating_tolerance_m", 2 * self.config.contact_roundoff_m)
        )
        self._pad_patch_tol_m = float(
            self.config.pad_patch_tol_m
            if self.config.pad_patch_tol_m is not None
            else getattr(self._gripper, "pad_patch_tolerance_m", self.config.contact_roundoff_m)
        )
        if self._pad_seating_tol_m < self.config.contact_roundoff_m:
            raise ValueError("resolved pad seating tolerance is smaller than numerical roundoff")
        if self._pad_patch_tol_m < self.config.contact_roundoff_m:
            raise ValueError("resolved pad patch tolerance is smaller than numerical roundoff")
        self.binding = digest(
            (
                assembly,
                self.config,
                self._provided,
                gripper_signature(self._gripper),
                self._pad_seating_tol_m,
                self._pad_patch_tol_m,
            )
        )

    def _catalogue(self, pid):
        if pid in self._catalogues:
            return self._catalogues[pid]
        cfg = self.config
        part = self._parts[pid]
        records = []
        seen = set()
        supplied = (
            [GraspRecord.from_dict(dict(g)) for g in self._provided[pid]]
            if pid in self._provided
            else None
        )
        candidates = generate_grasps(
            part,
            self._gripper,
            max_grasps=cfg.max_grasps,
            seed=cfg.seed,
            candidates=supplied,
            numerical_tol_m=cfg.contact_roundoff_m,
            pad_seating_tol_m=self._pad_seating_tol_m,
        )
        for grasp in candidates:
            key = digest((grasp.pose, grasp.jaw_width))
            if key in seen:
                continue
            seen.add(key)
            probe = self._gripper.clone()
            probe.grip_at(grasp.pose[:3, 3], grasp.pose[:3, :3], grasp.provenance["jaw_width"])
            links = probe.link_parts()
            centers = grasp_contacts(part, grasp, probe, self._backend, cfg.contact_roundoff_m * 2)
            contacts = finger_force_contacts(
                part, grasp, probe, self._backend, self._pad_patch_tol_m
            )
            closure = (
                dict(status="unknown", reason="pad_patch_unresolved")
                if contacts is None
                else check_force_closure(
                    contacts[0], contacts[1], friction=cfg.friction, cone_sides=cfg.cone_sides
                )
            )
            status = closure["status"]
            if status == "force_closed":
                for i, link in enumerate(links):
                    fv = (
                        link.geometry.vertices @ link.assembled_tf[:3, :3].T
                        + link.assembled_tf[:3, 3]
                    )
                    if (
                        i
                        and centers is not None
                        and any(
                            np.min((part.geometry.vertices - p) @ n) >= -self._pad_patch_tol_m
                            and np.max((fv - p) @ n) <= self._pad_patch_tol_m
                            for p, n in zip(*centers)
                        )
                    ):
                        continue
                    overlap = self._backend.classify_overlap(
                        part, np.eye(4), link, link.assembled_tf, budget=cfg.collision_budget
                    )
                    # 名义垫片之外的任何接触都不算合格，一律拒绝。
                    if overlap.status != "separated":
                        status = "unknown" if overlap.status == "unknown" else "self_collision"
                        break
            records.append((freeze(grasp.to_dict()), links, freeze(closure), status))
        self._catalogues[pid] = tuple(records)
        return self._catalogues[pid]

    def accepted_grasps(self, result: GraspabilityResult) -> tuple[GraspRecord, ...]:
        """该次查询所接受的抓取，重新构造为独立对象。"""
        if self._results.get(result.input_digest) is not result:
            raise ValueError("Unbound grasp result")
        pid = result.diagnostics["part_id"]
        return tuple(
            GraspRecord.from_dict(dict(self._catalogue(pid)[i][0])) for i in result.accepted_ids
        )

    def analyze(self, state: AssemblyState, part_id: str) -> GraspabilityResult:
        """待装零件必须已经以目标位姿出现在 `state` 中。"""
        if part_id not in state.poses or not set(state.poses).issubset(self._parts):
            raise ValueError("Query contains an absent/unknown part")
        key = digest((self.binding, state, part_id))
        if key in self._results:
            return self._results[key]
        started = perf_counter()
        records = self._catalogue(part_id)
        accepted = []
        details = []
        cfg = self.config
        target_tf = state.poses[part_id]
        unresolved = False
        hits = 0
        for gid, (_, links, closure, status) in enumerate(records):
            if status == "force_closed":
                for oid, otf in state.poses.items():
                    if oid == part_id:
                        continue
                    cache_key = digest((part_id, gid, target_tf, oid, otf))
                    if cache_key in self._environment:
                        collision = self._environment[cache_key]
                        hits += 1
                    else:
                        collision = "separated"
                        obstacle = self._parts[oid]
                        ov = obstacle.geometry.vertices @ otf[:3, :3].T + otf[:3, 3]
                        for link in links:
                            tf = target_tf @ link.assembled_tf
                            v = link.geometry.vertices @ tf[:3, :3].T + tf[:3, 3]
                            if np.any(v.min(0) > ov.max(0) + cfg.contact_roundoff_m) or np.any(
                                ov.min(0) > v.max(0) + cfg.contact_roundoff_m
                            ):
                                continue
                            result = self._backend.classify_overlap(
                                link, tf, obstacle, otf, budget=cfg.collision_budget
                            )
                            if result.status != "separated":
                                collision = (
                                    "unknown"
                                    if result.status == "unknown"
                                    else "environment_collision"
                                )
                                break
                        self._environment[cache_key] = collision
                    if collision != "separated":
                        status = collision
                        break
            if status == "force_closed":
                accepted.append(gid)
            if status == "unknown":
                unresolved = True
            details.append(dict(grasp_id=gid, status=status, closure=closure))
        result = GraspabilityResult(
            "partial" if unresolved else "complete",
            None if unresolved else len(accepted),
            tuple(accepted),
            len(records),
            key,
            dict(
                part_id=part_id,
                candidates=details,
                elapsed_s=perf_counter() - started,
                environment_cache_hits=hits,
                known_count=len(accepted),
                scope="fixed_calibrated_catalogue; force_closure_and_goal_gripper_collision_only",
                generation_max_grasps=cfg.max_grasps,
                numerical_roundoff_m=cfg.contact_roundoff_m,
                pad_seating_tolerance_m=self._pad_seating_tol_m,
                pad_patch_tolerance_m=self._pad_patch_tol_m,
                robot_and_path_pending=True,
            ),
        )
        self._results[key] = result
        return result
