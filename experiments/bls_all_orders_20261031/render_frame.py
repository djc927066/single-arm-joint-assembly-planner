"""Panda3D offscreen pictures of the triangular frame on the work table with the Panthera arm (2026-11-01).

Four images (same camera): the assembled frame, and the frame with beam_AB / beam_BC / beam_CA
removed (= the state left after removing that beam first). Table, robot base pose and part goals
from scenes/triangular_frame6/task.json; arm at its six-zero home; style as jbls/scene.py.

    python -X utf8 -m experiments.bls_all_orders_20261031.render_frame
Output: experiments/bls_all_orders_20261031/_out/render/frame_{assembled,without_beam_AB,...}.png
"""
from __future__ import annotations

import json
import sys

from curobo_bridge.runtime import prepare

prepare()

import numpy as np  # noqa: E402

from .run import OUT, log  # noqa: E402

COLORS = {"post_A": (0.20, 0.56, 0.27), "post_B": (0.80, 0.24, 0.24), "post_C": (0.23, 0.35, 0.58),
          "beam_AB": (0.93, 0.56, 0.18), "beam_BC": (0.90, 0.76, 0.22), "beam_CA": (0.62, 0.36, 0.66),
          "base": (0.47, 0.58, 0.72)}
TABLE, GRID, ROBOT = (0.86, 0.82, 0.74), (0.95, 0.93, 0.89), (0.55, 0.57, 0.60)


def box(lo, hi):
    import trimesh
    return trimesh.creation.box(extents=np.asarray(hi) - np.asarray(lo),
                                transform=trimesh.transformations.translation_matrix((np.asarray(lo) + np.asarray(hi)) / 2))


def main():
    import trimesh
    from panda3d.core import (AmbientLight, DirectionalLight, Filename, PNMImage, Point3, Vec3, Vec4,
                              loadPrcFileData)
    from adapters.panthera import PantheraArmProvider
    from curobo_bridge.export_model import wrs_collision_parts
    from lazy_cost.scene import FRAME_DIR, load_scene

    task = json.loads((FRAME_DIR / "task.json").read_text(encoding="utf-8"))
    tb = task["table"]
    tc, ts = np.asarray(tb["table_center"], float), np.asarray(tb["table_size"], float)
    sc = load_scene("triangular_frame6")
    arm = PantheraArmProvider(base_pos=tb["robot_base_pos"], base_rotmat=tb["robot_base_rotmat"]).arm()
    arm.goto_given_conf(np.zeros(6))
    arm.change_jaw_width(float(arm.end_effector.jaw_range[1]))

    items = [("table", box(tc - ts / 2, tc + ts / 2), np.eye(4), TABLE)]
    top = tc[2] + ts[2] / 2
    step, w = 0.05, 0.0012                     # grid lines on the table top
    for x in np.arange(tc[0] - ts[0] / 2 + step, tc[0] + ts[0] / 2 - 1e-6, step):
        items.append(("grid", box((x - w, tc[1] - ts[1] / 2, top), (x + w, tc[1] + ts[1] / 2, top + 0.0004)), np.eye(4), GRID))
    for y in np.arange(tc[1] - ts[1] / 2 + step, tc[1] + ts[1] / 2 - 1e-6, step):
        items.append(("grid", box((tc[0] - ts[0] / 2, y - w, top), (tc[0] + ts[0] / 2, y + w, top + 0.0004)), np.eye(4), GRID))
    for name, _, link in wrs_collision_parts(arm, include_base=True):
        m = link.cmodel.trm_mesh
        items.append(("robot", trimesh.Trimesh(m.vertices, m.faces, process=False), np.asarray(link.gl_homomat), ROBOT))
    for pid in ("base",) + tuple(sc.ids):
        items.append((pid, sc.meshes[pid], sc.goal[pid], COLORS[pid]))

    loadPrcFileData("", "window-type offscreen\nwin-size 1600 1100\naudio-library-name null\n"
                        "framebuffer-multisample 1\nmultisamples 8")
    from direct.showbase.ShowBase import ShowBase
    from wrs.basis.data_adapter import npmat4_to_pdmat4, pdgeomndp_from_vvnf
    base = ShowBase(windowType="offscreen")
    base.disableMouse()
    base.setBackgroundColor(1, 1, 1, 1)
    base.camLens.setNearFar(0.01, 20.0)
    base.camLens.setFov(32)
    site = sc.goal["base"][:3, 3]
    look = Point3(*(0.5 * (site + tc) + np.array([-0.02, 0.0, 0.04])))
    base.cam.setPos(look + Vec3(0.95, -1.30, 0.98))
    base.cam.lookAt(look, Vec3(0, 0, 1))
    amb = AmbientLight("ambient")
    amb.setColor(Vec4(0.55, 0.55, 0.55, 1))
    base.render.setLight(base.render.attachNewNode(amb))
    for hpr, c in (((-35, -55, 0), 0.62), ((120, -40, 0), 0.25)):
        sun = DirectionalLight("sun")
        sun.setColor(Vec4(c, c, c, 1))
        np_ = base.render.attachNewNode(sun)
        np_.setHpr(*hpr)
        base.render.setLight(np_)
    nodes = {}
    for i, (name, mesh, pose, color) in enumerate(items):
        node = pdgeomndp_from_vvnf(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.vertex_normals, np.float32),
                                   np.asarray(mesh.faces, np.uint32), f"{name}_{i}")
        node.reparentTo(base.render)
        node.setMat(npmat4_to_pdmat4(np.asarray(pose, float)))
        node.setColor(*color, 1.0)
        node.setTwoSided(True)
        nodes.setdefault(name, []).append(node)
    out = OUT / "render"
    out.mkdir(parents=True, exist_ok=True)
    try:
        for tag, hidden in (("assembled", None), ("without_beam_AB", "beam_AB"), ("without_beam_BC", "beam_BC"),
                            ("without_beam_CA", "beam_CA")):
            for pid in ("beam_AB", "beam_BC", "beam_CA"):
                for n in nodes[pid]:
                    (n.hide if pid == hidden else n.show)()
            for _ in range(4):
                base.graphicsEngine.renderFrame()
            base.graphicsEngine.syncFrame()
            shot = PNMImage()
            if not base.win.getScreenshot(shot):
                raise RuntimeError("offscreen framebuffer could not be read")
            f = out / f"frame_{tag}.png"
            shot.write(Filename.fromOsSpecific(str(f.resolve())))
            log(f"-> {f}")
    finally:
        base.destroy()


if __name__ == "__main__":
    sys.exit(main())
