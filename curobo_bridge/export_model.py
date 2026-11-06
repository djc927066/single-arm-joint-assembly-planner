"""把 WRS1 的 Panthera-HT 单臂（含夹爪）导出为带碰撞网格的 URDF。

与 ``dual_sealp/cuRobotest/model_adapter.py`` 的区别：那边导出的是**仅运动学** URDF
（没有碰撞几何，只能做 IK）。这里额外导出每个碰撞 link 的网格，供 cuRobo
``RobotBuilder`` 用 MorphIt 拟合碰撞球。

几何一致性：不重写 WRS 的运动学或网格位姿。对每个碰撞 link，取 WRS 当前
``lnk.gl_homomat``（= ``cmodel.homomat``）相对它所属关节坐标系的变换，把网格顶点
直接变换进该关节坐标系后写出。所以导出的三角形与 WRS 在 q=0 时逐点一致；
其余关节角下的一致性由 :func:`verify_export` 用随机关节角实测。

WRS 的外部碰撞集合（``PantheraHTSglArm.setup_cc``）只有 link2..link5、掌部和两指；
base_link 与 link1 不参与外部碰撞，link6 网格被禁用。``include_base=True``
时额外导出 base_link/link1，用于 cuRobo 自碰撞或更完整的安全检查。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
ARM_LINKS = [f"link{i}" for i in range(1, 7)]
#: Panthera-HT_description_with_finger.urdf 的关节速度上限 (rad/s)。
URDF_VELOCITY_LIMITS = [4.2, 5.0, 5.0, 4.2, 3.7, 3.7]
#: WRS 外部碰撞检测覆盖的部件（与 PantheraHTSglArm.setup_cc 一致）。
WRS_EXTERNAL_LINKS = ("link2", "link3", "link4", "link5",
                      "gripper_palm", "finger_left", "finger_right")
GRIPPER_LINKS = ("gripper_palm", "finger_left", "finger_right")
FINGER_LINKS = ("finger_left", "finger_right")
TOOL_FRAME = "wrs_tcp"


def _homomat(pos, rotmat) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = np.asarray(rotmat, dtype=float)
    out[:3, 3] = np.asarray(pos, dtype=float)
    return out


@dataclass
class ExportedPanthera:
    out_dir: Path
    urdf_path: Path
    chain_path: Path
    jaw_width: float
    collision_links: tuple
    base_homomat: np.ndarray
    home: np.ndarray
    limits: np.ndarray
    velocity_limits: np.ndarray
    wrs_part_of: dict = field(default_factory=dict)


def make_wrs_arm(base_pos=np.zeros(3), base_rotmat=np.eye(3), jaw_width=0.085,
                 enable_cc=True):
    from wrs.robot_sim.robots.robot_panthera_ht.panthera_ht import PantheraHTSglArm

    arm = PantheraHTSglArm(pos=np.asarray(base_pos, float),
                           rotmat=np.asarray(base_rotmat, float),
                           enable_cc=enable_cc)
    arm.change_jaw_width(float(jaw_width))
    arm.goto_given_conf(np.zeros(6))
    return arm


def wrs_collision_parts(arm, include_base: bool):
    """[(urdf_link_name, parent_frame_index, wrs_link)]; parent -1 = base frame."""
    manip, ee = arm.manipulator, arm.end_effector
    jnts = manip.jlc.jnts
    parts = []
    if include_base:
        parts.append(("base_link", -1, manip.jlc.anchor.lnk_list[0]))
        parts.append(("link1", 0, jnts[0].lnk))
    for i in range(1, 5):
        parts.append((f"link{i + 1}", i, jnts[i].lnk))
    parts.append(("gripper_palm", 5, ee.jlc.anchor.lnk_list[0]))
    parts.append(("finger_left", 5, ee.jlc.jnts[0].lnk))
    parts.append(("finger_right", 5, ee.jlc.jnts[1].lnk))
    return [p for p in parts if p[2].cmodel is not None]


def _frames(arm):
    jlc = arm.manipulator.jlc
    base_p, base_r = jlc.anchor.gl_flange_pose_list[0]
    base = _homomat(base_p, base_r)
    return base, [np.asarray(j.gl_homomat_q, float) for j in jlc.jnts]


def _urdf_joint(robot, name, parent, child, transform, kind="fixed", axis=None,
                limits=None, velocity=None):
    item = ET.SubElement(robot, "joint", name=name, type=kind)
    ET.SubElement(item, "parent", link=parent)
    ET.SubElement(item, "child", link=child)
    transform = np.asarray(transform, float)
    xyz = " ".join(f"{x:.17g}" for x in transform[:3, 3])
    rpy = " ".join(f"{x:.17g}" for x in Rotation.from_matrix(transform[:3, :3]).as_euler("xyz"))
    ET.SubElement(item, "origin", xyz=xyz, rpy=rpy)
    if axis is not None:
        ET.SubElement(item, "axis", xyz=" ".join(f"{a:.17g}" for a in axis))
        ET.SubElement(item, "limit", lower=f"{limits[0]:.17g}", upper=f"{limits[1]:.17g}",
                      effort="10", velocity=f"{velocity:.17g}")


def _urdf_link(robot, name, mesh_rel=None):
    link = ET.SubElement(robot, "link", name=name)
    if mesh_rel is not None:
        for tag in ("visual", "collision"):
            node = ET.SubElement(link, tag)
            ET.SubElement(node, "origin", xyz="0 0 0", rpy="0 0 0")
            geom = ET.SubElement(node, "geometry")
            ET.SubElement(geom, "mesh", filename=mesh_rel)
    return link


def export_panthera(out_dir, *, base_pos=np.zeros(3), base_rotmat=np.eye(3),
                    jaw_width=0.085, include_base=False, arm=None) -> ExportedPanthera:
    """Write ``panthera.urdf``, ``meshes/*.stl`` and ``chain.json`` into ``out_dir``."""
    import trimesh

    out_dir = Path(out_dir)
    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)
    arm = arm or make_wrs_arm(base_pos, base_rotmat, jaw_width)
    arm.goto_given_conf(np.zeros(6))
    base, frames = _frames(arm)
    jnts = arm.manipulator.jlc.jnts

    meshes = {}
    wrs_part_of = {}
    for name, parent, lnk in wrs_collision_parts(arm, include_base):
        ref = base if parent < 0 else frames[parent]
        rel = np.linalg.inv(ref) @ np.asarray(lnk.gl_homomat, float)
        src = lnk.cmodel.trm_mesh
        verts = np.asarray(src.vertices, float) @ rel[:3, :3].T + rel[:3, 3]
        mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(src.faces), process=False)
        rel_path = f"meshes/{name}.stl"
        mesh.export(out_dir / rel_path)
        meshes[name] = rel_path
        wrs_part_of[name] = getattr(lnk.cmodel, "name", name)

    tcp = _homomat(arm.manipulator.gl_tcp_pos, arm.manipulator.gl_tcp_rotmat)
    tcp_rel = np.linalg.inv(frames[5]) @ tcp

    robot = ET.Element("robot", name="wrs_panthera_ht_collision")
    _urdf_link(robot, "world")
    _urdf_link(robot, "base_link", meshes.get("base_link"))
    _urdf_joint(robot, "world_to_base", "world", "base_link", base)
    parent = "base_link"
    limits, axes, origins = [], [], []
    for i, jnt in enumerate(jnts):
        child = ARM_LINKS[i]
        origin = _homomat(jnt.loc_pos, jnt.loc_rotmat)
        rng = np.asarray(jnt.motion_range, float)
        _urdf_link(robot, child, meshes.get(child))
        _urdf_joint(robot, JOINT_NAMES[i], parent, child, origin, "revolute",
                    np.asarray(jnt.loc_motion_ax, float), rng, URDF_VELOCITY_LIMITS[i])
        limits.append(rng.tolist())
        axes.append(np.asarray(jnt.loc_motion_ax, float).tolist())
        origins.append(origin.tolist())
        parent = child
    for name in GRIPPER_LINKS:
        if name in meshes:
            _urdf_link(robot, name, meshes[name])
            _urdf_joint(robot, f"link6_to_{name}", "link6", name, np.eye(4))
    _urdf_link(robot, TOOL_FRAME)
    _urdf_joint(robot, "link6_to_wrs_tcp", "link6", TOOL_FRAME, tcp_rel)

    urdf_path = out_dir / "panthera.urdf"
    ET.indent(robot)
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)

    home = np.asarray(arm.manipulator.home_conf, float)
    chain = dict(
        schema_version="wrs_panthera_collision_v1",
        joint_names=JOINT_NAMES, base=base.tolist(), tcp=tcp_rel.tolist(),
        origins=origins, axes=axes, limits=limits,
        velocity_limits=URDF_VELOCITY_LIMITS, home=home.tolist(),
        jaw_width=float(jaw_width), collision_links=list(meshes),
        wrs_external_links=list(WRS_EXTERNAL_LINKS), include_base=bool(include_base))
    chain_path = out_dir / "chain.json"
    chain_path.write_text(json.dumps(chain, indent=2), encoding="utf-8")
    return ExportedPanthera(out_dir=out_dir, urdf_path=urdf_path, chain_path=chain_path,
                            jaw_width=float(jaw_width), collision_links=tuple(meshes),
                            base_homomat=base, home=home, limits=np.asarray(limits),
                            velocity_limits=np.asarray(URDF_VELOCITY_LIMITS),
                            wrs_part_of=wrs_part_of)


def conservative_envelope(mesh, *, pitch=0.0015, dilation_voxels=1, max_faces=6000):
    """Watertight solid that contains ``mesh``, for MorphIt sphere fitting.

    Panthera CAD meshes are mostly non-watertight multi-shell exports (up to
    320k faces), so inside/outside tests - which MorphIt and its metrics rely
    on - are undefined on them. Surface voxels (open3d) are closed, hole-filled
    and dilated by ``dilation_voxels`` before marching cubes, then decimated.
    Measured on all nine links: every original surface sample lies inside the
    envelope; the envelope surface is 1.5-3 mm outside the original (similar to
    WRS's 2 mm primitive padding).
    """
    import open3d as o3d
    import trimesh
    from scipy import ndimage
    from skimage import measure

    if dilation_voxels < 1:
        # scipy treats iterations=0 as "dilate until nothing changes".
        raise ValueError("dilation_voxels must be >= 1")
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
                                   o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
    margin = (dilation_voxels + 3) * pitch
    lo = np.asarray(mesh.bounds[0]) - margin
    hi = np.asarray(mesh.bounds[1]) + margin
    grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(om, pitch, lo, hi)
    idx = np.array([v.grid_index for v in grid.get_voxels()])
    occ = np.zeros(np.ceil((hi - lo) / pitch).astype(int) + 2, bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    solid = ndimage.binary_fill_holes(ndimage.binary_closing(occ, iterations=1) | occ)
    solid = ndimage.binary_dilation(solid, iterations=int(dilation_voxels))
    verts, faces, _, _ = measure.marching_cubes(np.pad(solid, 1).astype(np.float32), 0.5)
    verts = (verts - 0.5) * pitch + lo      # padded voxel-centre index -> world
    raw = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                    o3d.utility.Vector3iVector(faces))
    raw = raw.simplify_quadric_decimation(target_number_of_triangles=int(max_faces))
    out = trimesh.Trimesh(np.asarray(raw.vertices), np.asarray(raw.triangles))
    trimesh.repair.fix_normals(out)
    return out


def write_fit_urdf(exported: ExportedPanthera, *, pitch=0.0015, dilation_voxels=1,
                   max_faces=6000) -> Path:
    """Write ``panthera_fit.urdf`` whose link meshes are conservative envelopes.

    The kinematic tree is identical to ``panthera.urdf``; only mesh filenames
    change to ``meshes_fit/*.stl``. Use it for ``RobotBuilder`` only; the
    original meshes stay the geometric reference.
    """
    import trimesh

    fit_dir = exported.out_dir / "meshes_fit"
    fit_dir.mkdir(exist_ok=True)
    report = {}
    for name in exported.collision_links:
        src = trimesh.load(exported.out_dir / f"meshes/{name}.stl")
        env = conservative_envelope(src, pitch=pitch, dilation_voxels=dilation_voxels,
                                    max_faces=max_faces)
        env.export(fit_dir / f"{name}.stl")
        report[name] = dict(faces=int(len(env.faces)), watertight=bool(env.is_watertight))
    text = exported.urdf_path.read_text(encoding="utf-8").replace(
        'filename="meshes/', 'filename="meshes_fit/')
    out = exported.out_dir / "panthera_fit.urdf"
    out.write_text(text, encoding="utf-8")
    (exported.out_dir / "meshes_fit" / "envelope_report.json").write_text(
        json.dumps(dict(pitch=pitch, dilation_voxels=dilation_voxels, links=report), indent=2),
        encoding="utf-8")
    return out


def load_chain(out_dir) -> dict:
    return json.loads((Path(out_dir) / "chain.json").read_text(encoding="utf-8"))


def fk_frames(chain: dict, q) -> list:
    """World pose of link1..link6 (joint frames) for the exported chain."""
    t = np.asarray(chain["base"], float).copy()
    out = []
    for origin, axis, angle in zip(chain["origins"], chain["axes"], q):
        rot = _homomat(np.zeros(3), Rotation.from_rotvec(np.asarray(axis) * angle).as_matrix())
        t = t @ np.asarray(origin) @ rot
        out.append(t.copy())
    return out


def verify_export(exported: ExportedPanthera, arm, *, samples=64, seed=0) -> dict:
    """Compare exported chain (frames + mesh vertices + TCP) with WRS at random q."""
    import trimesh

    chain = json.loads(exported.chain_path.read_text(encoding="utf-8"))
    parts = {n: (p, l) for n, p, l in wrs_collision_parts(
        arm, "base_link" in exported.collision_links)}
    # STL stores unindexed triangles, so compare triangle corners (face order is
    # preserved by the STL writer), never vertex indices.
    local = {}
    for n in exported.collision_links:
        m = trimesh.load(exported.out_dir / f"meshes/{n}.stl", process=False)
        local[n] = np.asarray(m.vertices, float)[np.asarray(m.faces)].reshape(-1, 3)
    rng = np.random.default_rng(seed)
    lim = exported.limits
    worst_vertex, worst_tcp = 0.0, 0.0
    for q in rng.uniform(lim[:, 0], lim[:, 1], size=(samples, 6)):
        arm.goto_given_conf(q)
        frames = fk_frames(chain, q)
        base = np.asarray(chain["base"], float)
        for name, (parent, lnk) in parts.items():
            ref = base if parent < 0 else frames[parent]
            tm = lnk.cmodel.trm_mesh
            src = np.asarray(tm.vertices, float)[np.asarray(tm.faces)].reshape(-1, 3)
            idx = rng.choice(len(src), size=min(3000, len(src)), replace=False)
            ours = local[name][idx] @ ref[:3, :3].T + ref[:3, 3]
            wrs_h = np.asarray(lnk.gl_homomat, float)
            theirs = src[idx] @ wrs_h[:3, :3].T + wrs_h[:3, 3]
            worst_vertex = max(worst_vertex,
                               float(np.max(np.linalg.norm(ours - theirs, axis=1))))
        tcp = frames[5] @ np.asarray(chain["tcp"], float)
        p, _ = arm.fk(q)
        worst_tcp = max(worst_tcp, float(np.linalg.norm(tcp[:3, 3] - p)))
    arm.goto_given_conf(np.zeros(6))
    return dict(samples=samples, max_vertex_error_m=worst_vertex, max_tcp_error_m=worst_tcp)
