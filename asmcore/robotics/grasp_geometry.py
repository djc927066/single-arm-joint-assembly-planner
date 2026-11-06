"""抓取接触几何与候选生成。

逐字迁移自 WRS3 `wrs/assembly/robotics/execution.py` 的
`_box_grasps` / `generate_execution_grasps` / `_grasp_contacts` /
`_finger_force_contacts`。数值、容差、判据一律未改，只把取用夹爪的方式换成
`GripperPort`：

    part_from_scene_object(link, …)  ->  gripper.link_parts()
    rigid_tf_from_wrs(link.tf)       ->  part.assembled_tf
    wrs.grasp.grasp.Grasp            ->  GraspRecord

**坐标系约定**：抓取位姿表达在零件局部系，因此 `grip_at` 收到的是零件系位姿，
`link_parts()` 返回的 `assembled_tf` 也在零件系。上层再用 `target_tf @ tf`
换到世界系——与原版完全一致。
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from ..contact.analysis import analyze_pair
from ..geometry.proximity import MeshProximity, ProximityBackend
from ..mechanics.force_points import prepare_force_points
from ..model import ContactConfig, FloatArray, IntArray, MeshData, Part
from .grasp_model import GraspRecord, GripperPort, finger_parts


def box_grasps(part: Part, gripper: GripperPort) -> list[GraspRecord]:
    """经过校验的轴对齐长方体的四个中心侧夹。确定性，无随机。"""
    v = part.geometry.vertices
    lo, hi = v.min(0), v.max(0)
    size = hi - lo
    corners = np.all((np.abs(v - lo) < 1e-9) | (np.abs(v - hi) < 1e-9), axis=1)
    if len(np.unique(v, axis=0)) != 8 or len(part.geometry.faces) != 12 or not np.all(corners):
        return []
    result = []
    for angle in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
        r = Rotation.from_euler("z", angle).as_matrix() @ np.diag([1.0, -1.0, -1.0])
        opening = r @ np.asarray(gripper.open_dir, dtype=float)
        width = float(np.abs(opening) @ size)
        if not gripper.jaw_range[0] <= width <= gripper.jaw_range[1]:
            continue
        center = (lo + hi) / 2
        center[2] += 0.2 * size[2]
        pose = np.eye(4)
        pose[:3, :3] = r
        pose[:3, 3] = center
        pre = pose.copy()
        pre[:3, 3] -= r[:, 2] * 0.06
        contacts = np.array([center - opening * width / 2, center + opening * width / 2])
        result.append(
            GraspRecord(
                pose, pre, width,
                provenance={
                    "jaw_width": width,
                    "contacts_local_m": contacts,
                    "normals_on_object_local": np.array([opening, -opening]),
                    "generator": "deterministic_box_side_pinch",
                },
            )
        )
    return result


def generate_grasps(
    part: Part,
    gripper: GripperPort,
    *,
    max_grasps: int = 12,
    seed: int = 7,
    candidates: Iterable[GraspRecord] | None = None,
    numerical_tol_m: float = 2e-6,
    pad_seating_tol_m: float | None = None,
) -> tuple[GraspRecord, ...]:
    """候选抓取：长方体走确定性侧夹，其余交给夹爪自带的候选生成。

    名义开口不等于 STL 上的实际贴合距离（OR2FG7 每侧有 0.5 mm 垫片偏置），
    因此先标定、再在网格上复核——这一段与原版逐行一致。
    """
    grasps = list(candidates) if candidates is not None else box_grasps(part, gripper)
    if not grasps and candidates is None:
        maker = getattr(gripper, "antipodal_candidates", None)
        if maker is not None:
            grasps = list(maker(part, max_grasps=max_grasps, seed=seed))[:max_grasps]
    if pad_seating_tol_m is None:
        pad_seating_tol_m = 2.0 * numerical_tol_m
    if numerical_tol_m <= 0 or pad_seating_tol_m < numerical_tol_m:
        raise ValueError("invalid grasp geometry tolerances")
    probe = gripper.clone()
    backend = MeshProximity(numerical_tol_m=numerical_tol_m)
    accepted: list[GraspRecord] = []
    for g in grasps[:max_grasps]:
        contacts = grasp_contacts(part, g, gripper, backend, 2.0 * numerical_tol_m)
        if contacts is None:
            continue
        points, inward = contacts
        width = g.jaw_width

        def pad_query(command):
            probe.grip_at(g.pose[:3, 3], g.pose[:3, :3], command)
            distances, witnesses = [], []
            for fp in finger_parts(probe):
                tf = fp.assembled_tf
                query = backend.closest_points(fp.geometry, (points - tf[:3, 3]) @ tf[:3, :3])
                if query.status != "complete":
                    return None
                distances.append(query.distances_m)
                witnesses.append(query.points_local_m @ tf[:3, :3].T + tf[:3, 3])
            choice = np.argmin(distances, axis=0)
            nearest = np.asarray(witnesses)[choice, np.arange(2)]
            return nearest, np.min(distances, axis=0)

        measured = pad_query(width)
        if measured is None:
            continue
        nearest, distances = measured
        gap = np.einsum("ij,ij->i", nearest - points, -inward)
        if np.max(np.abs(distances - gap)) > pad_seating_tol_m:
            continue
        command = float(width - 2 * np.mean(gap))
        if not probe.jaw_range[0] <= command <= probe.jaw_range[1]:
            continue
        verified = pad_query(command)
        if verified is None or np.max(verified[1]) > pad_seating_tol_m:
            continue
        accepted.append(
            g.replace(
                jaw_width=command,
                provenance=dict(
                    jaw_width=command,
                    contacts_local_m=points,
                    normals_on_object_local=inward,
                    pad_residual_m=float(np.max(verified[1])),
                    pad_seating_tolerance_m=float(pad_seating_tol_m),
                    nominal_width_before_calibration_m=width,
                ),
            )
        )
    return tuple(accepted)


def grasp_contacts(
    part: Part,
    grasp: GraspRecord,
    gripper: GripperPort,
    backend: ProximityBackend,
    tolerance: float,
) -> tuple[FloatArray, FloatArray] | None:
    """两个名义接触点及其指向零件内部的法向；面不对置则判为无效。"""
    c = np.asarray(grasp.pose[:3, 3], dtype=float)
    axis = np.asarray(grasp.pose[:3, :3], dtype=float) @ np.asarray(gripper.open_dir, dtype=float)
    axis /= np.linalg.norm(axis)
    width = grasp.provenance.get("jaw_width", grasp.jaw_width)
    if width is None:
        return None
    points = np.asarray(
        grasp.provenance.get("contacts_local_m", [c - axis * width / 2, c + axis * width / 2]),
        dtype=float,
    )
    if points.shape != (2, 3):
        return None
    q = backend.closest_points(part.geometry, points)
    if q.status != "complete" or np.max(q.distances_m) > tolerance:
        return None
    tri = part.geometry.vertices[part.geometry.faces[q.face_ids]]
    outward = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    outward /= np.linalg.norm(outward, axis=1)[:, None]
    inward = np.array([axis, -axis])
    if np.max(np.einsum("ij,ij->i", outward, inward)) > -0.99:
        return None
    return q.points_local_m, inward


def finger_force_contacts(
    part: Part,
    grasp: GraspRecord,
    gripper: GripperPort,
    backend: ProximityBackend,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, IntArray] | None:
    """实际的垫片/零件重叠多边形；**每根手指共享一份力容量**。"""
    probe = gripper.clone()
    probe.grip_at(grasp.pose[:3, 3], grasp.pose[:3, :3], grasp.provenance["jaw_width"])
    centers = grasp_contacts(part, grasp, gripper, backend, tolerance * 2)
    if centers is None:
        return None
    points, normals, groups = [], [], []
    cfg = ContactConfig(max_triangle_tests=200000, max_cells=1000)
    for index, finger in enumerate(finger_parts(probe)):
        ftf = finger.assembled_tf
        fvertices = finger.geometry.vertices @ ftf[:3, :3].T + ftf[:3, 3]
        for center, inward in zip(*centers):
            # 取目标的承载面片与手指上真正共面的垫片三角形。整根手指的穿透
            # 另行检查；它的安装法兰不必落在垫片半空间里。
            if np.min((part.geometry.vertices - center) @ inward) < -tolerance:
                continue
            patches = []
            for vertices, faces, pid in (
                (part.geometry.vertices, part.geometry.faces, "target_pad"),
                (fvertices, finger.geometry.faces, "finger_pad"),
            ):
                tri = vertices[faces]
                selected = tri[np.max(np.abs((tri - center) @ inward), axis=1) <= tolerance]
                if not len(selected):
                    break
                # 显式的浮点就位误差，受 tolerance 约束；绝不把任意的 SDF
                # 近接带提升为承力面。
                selected = selected - ((selected - center) @ inward)[..., None] * inward
                vertices, indices = np.unique(selected.reshape(-1, 3), axis=0, return_inverse=True)
                mesh = MeshData(vertices, indices.reshape(-1, 3), orientation="trusted")
                patches.append(Part(pid, mesh, np.eye(4)))
            if len(patches) != 2:
                continue
            report = analyze_pair(
                patches[0], np.eye(4), patches[1], np.eye(4), config=cfg, backend=backend
            )
            for patch in report.patches:
                if patch.classification != "active" or patch.quality not in ("analytic", "bounded"):
                    continue
                prepared, _ = prepare_force_points(patch)
                points.extend(prepared.points)
                normals.extend(-prepared.normals)
                groups.extend([index] * len(prepared.points))
    if len(set(groups)) < 2:
        return None
    return np.asarray(points), np.asarray(normals), np.asarray(groups, dtype=int)
