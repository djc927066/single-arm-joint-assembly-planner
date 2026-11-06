"""随机桌面装配场景：同一份几何同时生成 WRS 障碍与 cuRobo ``Scene``。

布置与 ``costlab/arm_rrt_probe.py`` 一致：Panthera 基座在世界原点，桌面上表面 z=0，
桌板中心 (0.33, 0, -0.01)、尺寸 0.64 x 0.60 x 0.02 m。

障碍 = 桌板 + 若干真实零件网格（sealp 资产里的封闭网格，含 U 形、十字、圆柱等非凸件，
随机偏航、部分侧放）+ 若干悬空盒子（用来打到中段连杆）。

WRS 侧：零件用 ``mcm.CollisionModel(stl)``。注意 sealp 的 ``make_collision_model`` 传入的
``cdprim_type="triangles"`` 在 WRS1 中不是合法枚举，会静默回退到默认 AABB，因此这里
直接用默认构造，与 sealp 实际运行时的语义一致（局部 AABB + 2 mm）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .runtime import REPO_ROOT

TABLE_CENTRE = np.array([0.33, 0.0, -0.01])
TABLE_SIZE = np.array([0.64, 0.60, 0.02])
MODEL_ROOT = REPO_ROOT / "sealp" / "assets" / "models"
PART_MESHES = (
    "CrossRailFrameV1/model/rail.stl",
    "Toy/model/post.stl",
    "Toy/model/top_cross.stl",
    "first/part_6.stl",
    "first/part_8.stl",
    "second/ao.stl",
    "second/cylinder.stl",
    "second/tu.stl",
    "yuanchair/yuanchair-part2.stl",
)


@dataclass
class Obstacle:
    name: str
    kind: str                      # "box" | "mesh"
    pose: np.ndarray               # 4x4 world pose
    dims: np.ndarray | None = None  # box full extents
    mesh_path: Path | None = None


@dataclass
class TableScene:
    name: str
    obstacles: list = field(default_factory=list)

    # ---- WRS -------------------------------------------------------
    def wrs_obstacles(self):
        import wrs.modeling.collision_model as mcm

        out = []
        for ob in self.obstacles:
            if ob.kind == "box":
                cm = mcm.gen_box(xyz_lengths=np.asarray(ob.dims, float),
                                 pos=ob.pose[:3, 3], rotmat=ob.pose[:3, :3])
            else:
                cm = mcm.CollisionModel(initor=str(ob.mesh_path), name=ob.name)
                cm.pose = (ob.pose[:3, 3].copy(), ob.pose[:3, :3].copy())
            out.append(cm)
        return out

    # ---- cuRobo ----------------------------------------------------
    def curobo_scene(self):
        import trimesh
        from curobo.scene import Cuboid, Mesh, Scene

        cuboids, meshes = [], []
        for ob in self.obstacles:
            pose = pose7(ob.pose)
            if ob.kind == "box":
                cuboids.append(Cuboid(name=ob.name, dims=list(map(float, ob.dims)), pose=pose))
            else:
                m = trimesh.load(ob.mesh_path, process=True)
                meshes.append(Mesh(name=ob.name, vertices=np.asarray(m.vertices).tolist(),
                                   faces=np.asarray(m.faces).reshape(-1).tolist(), pose=pose))
        return Scene(cuboid=cuboids, mesh=meshes)

    def curobo_scene_dict(self) -> dict:
        """Dict form accepted by ``SceneCfg.create`` and ``MotionPlannerCfg.create``.

        A list of these gives one collision world per environment (multi_env).
        """
        import trimesh

        out = {"cuboid": {}, "mesh": {}}
        for ob in self.obstacles:
            if ob.kind == "box":
                out["cuboid"][ob.name] = dict(dims=list(map(float, ob.dims)), pose=pose7(ob.pose))
            else:
                m = trimesh.load(ob.mesh_path, process=True)
                out["mesh"][ob.name] = dict(vertices=np.asarray(m.vertices).tolist(),
                                            faces=np.asarray(m.faces).reshape(-1).tolist(),
                                            pose=pose7(ob.pose))
        return out

    def trimeshes(self):
        """World-frame trimesh of every obstacle (reference geometry)."""
        import trimesh

        out = []
        for ob in self.obstacles:
            if ob.kind == "box":
                m = trimesh.creation.box(extents=ob.dims)
            else:
                m = trimesh.load(ob.mesh_path, process=True)
            m = m.copy()
            m.apply_transform(ob.pose)
            out.append(m)
        return out

    def counts(self):
        return dict(boxes=sum(o.kind == "box" for o in self.obstacles),
                    meshes=sum(o.kind == "mesh" for o in self.obstacles))


def pose7(h) -> list:
    """4x4 -> [x, y, z, qw, qx, qy, qz] (cuRobo convention)."""
    q = Rotation.from_matrix(np.asarray(h)[:3, :3]).as_quat()
    return [float(h[0, 3]), float(h[1, 3]), float(h[2, 3]),
            float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def _homomat(pos, rotmat):
    h = np.eye(4)
    h[:3, :3], h[:3, 3] = rotmat, pos
    return h


def random_scene(seed: int, *, n_parts: int = 6, n_floating: int = 2) -> TableScene:
    import trimesh

    rng = np.random.default_rng(seed)
    scene = TableScene(name=f"table_s{seed}")
    scene.obstacles.append(Obstacle("work_table", "box", _homomat(TABLE_CENTRE, np.eye(3)),
                                    dims=TABLE_SIZE.copy()))
    for k in range(n_parts):
        rel = PART_MESHES[int(rng.integers(len(PART_MESHES)))]
        path = MODEL_ROOT / rel
        m = trimesh.load(path, process=True)
        rot = Rotation.from_euler("z", rng.uniform(-np.pi, np.pi))
        if rng.random() < 0.3:
            rot = rot * Rotation.from_euler("x", np.pi / 2)
        rotmat = rot.as_matrix()
        lo = (m.vertices @ rotmat.T)[:, 2].min()
        for _ in range(50):
            xy = np.array([rng.uniform(0.10, 0.60), rng.uniform(-0.27, 0.27)])
            if not (xy[0] < 0.14 and abs(xy[1]) < 0.12):   # keep off the robot base
                break
        pos = np.array([xy[0], xy[1], -lo + 1e-4])
        scene.obstacles.append(Obstacle(f"part{k}_{Path(rel).stem}", "mesh",
                                        _homomat(pos, rotmat), mesh_path=path))
    for k in range(n_floating):
        dims = rng.uniform(0.03, 0.10, size=3)
        pos = np.array([rng.uniform(0.10, 0.55), rng.uniform(-0.30, 0.30),
                        rng.uniform(0.08, 0.45)])
        rotmat = Rotation.from_euler("z", rng.uniform(-np.pi, np.pi)).as_matrix()
        scene.obstacles.append(Obstacle(f"float{k}", "box", _homomat(pos, rotmat), dims=dims))
    return scene
