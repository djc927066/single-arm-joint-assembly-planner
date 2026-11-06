"""How the optimistic value of an arrow is computed, drawn in 3D on the frame (chain cost) (2026-11-01).

From _out/worked_bound_triangular_frame6.json (worked_bound.py). One picture per arrow of the
relaxed-best order A -> B -> AB -> C -> BC -> CA, and one pruned arrow ({A,B,C,CA} --BC->):
installed set S at its goals, the part p at its cheapest pose of D(S, p) on the table (solid) and at
its goal (ghost), and the moves the step cost measures (part origin as end-effector proxy):
  transit  above goal(last) (or HOME) -> above pick   blue
  pick     above pick -> pick (counted 0.10)           dark
  transfer above pick -> above goal(p)                 magenta
  insert   above goal(p) -> goal(p) (counted 0.10)     dark
Then the relaxed layout (every part at its own cheapest pose: they may overlap, the relaxation
ignores it) and a bar chart F + step + H per arrow against U.

    python -X utf8 -m experiments.bls_all_orders_20261031.viz_bound_steps
"""
from __future__ import annotations

import json
import sys

from curobo_bridge.runtime import prepare

prepare()

import numpy as np  # noqa: E402

from .run import LIFT, OUT, log  # noqa: E402
from .viz_domain import COLORS, SHORT  # noqa: E402

PATH = ["post_A", "post_B", "beam_AB", "post_C", "beam_BC", "beam_CA"]
PRUNED = (("beam_CA", "post_A", "post_B", "post_C"), "beam_BC")
TRANSIT, TRANSFER, VERT = (0.16, 0.47, 0.84), (0.84, 0.23, 0.60), (0.15, 0.15, 0.15)
U = 3.0509


def main():
    import trimesh
    from panda3d.core import (AmbientLight, DirectionalLight, Filename, LineSegs, PNMImage, Point3,
                              TransparencyAttrib, Vec3, Vec4, loadPrcFileData)
    from adapters.panthera import PantheraArmProvider
    from curobo_bridge.export_model import wrs_collision_parts
    from joint_search.problem import Problem
    from lazy_cost.scene import FRAME_DIR
    from .render_frame import GRID, ROBOT, TABLE, box

    d = json.loads((OUT / "worked_bound_triangular_frame6.json").read_text(encoding="utf-8"))
    rows = {(tuple(sorted(r["S"])), r["p"]): r for r in d["rows"]}
    home = np.asarray(d["home"])
    goal = {p: np.asarray(v) for p, v in d["goal"].items()}
    up = np.array([0, 0, LIFT])
    pb = Problem("triangular_frame6", center=(0.373, 0.0, 0.0))
    sc = pb.scene
    tb = json.loads((FRAME_DIR / "task.json").read_text(encoding="utf-8"))["table"]
    tc, ts = np.asarray(sc.table_center, float), np.asarray(sc.table_size, float)
    arm = PantheraArmProvider(base_pos=tb["robot_base_pos"], base_rotmat=tb["robot_base_rotmat"]).arm()
    arm.goto_given_conf(np.zeros(6))
    arm.change_jaw_width(float(arm.end_effector.jaw_range[1]))
    links = wrs_collision_parts(arm, include_base=True)

    steps = []                                   # (S, p, chosen candidate row)
    S, last = (), None
    for p in PATH:
        r = rows[tuple(sorted(S)), p]
        c = next(c for c in r["cands"] if c["last"] == last)
        steps.append((tuple(sorted(S)), p, c))
        S, last = S + (p,), p
    r = rows[tuple(sorted(PRUNED[0])), PRUNED[1]]
    steps.append((tuple(sorted(PRUNED[0])), PRUNED[1], min(r["cands"], key=lambda c: c["total"])))
    for S, p, c in steps:
        assert abs(c["F"] + c["step"] + c["H"] - c["total"]) < 1e-9

    loadPrcFileData("", "window-type offscreen\nwin-size 1400 1000\naudio-library-name null\n"
                        "framebuffer-multisample 1\nmultisamples 8")
    from direct.showbase.ShowBase import ShowBase
    from wrs.basis.data_adapter import npmat4_to_pdmat4, pdgeomndp_from_vvnf
    base = ShowBase(windowType="offscreen")
    base.disableMouse()
    base.setBackgroundColor(1, 1, 1, 1)
    base.camLens.setNearFar(0.01, 20.0)
    base.camLens.setFov(30)
    amb = AmbientLight("ambient")
    amb.setColor(Vec4(0.55, 0.55, 0.55, 1))
    base.render.setLight(base.render.attachNewNode(amb))
    for hpr, c in (((-35, -55, 0), 0.62), ((120, -40, 0), 0.25)):
        sun = DirectionalLight("sun")
        sun.setColor(Vec4(c, c, c, 1))
        n_ = base.render.attachNewNode(sun)
        n_.setHpr(*hpr)
        base.render.setLight(n_)
    live = []

    def add(mesh, pose, color, alpha=1.0):
        node = pdgeomndp_from_vvnf(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.vertex_normals, np.float32),
                                   np.asarray(mesh.faces, np.uint32), f"n{len(live)}")
        node.reparentTo(base.render)
        node.setMat(npmat4_to_pdmat4(np.asarray(pose, float)))
        node.setColor(*color, alpha)
        node.setTwoSided(True)
        if alpha < 1:
            node.setTransparency(TransparencyAttrib.MAlpha)
            node.setDepthWrite(False)
            node.setBin("fixed", 50)
        live.append(node)

    def line(a, b, color, w=5.0):
        ls = LineSegs()
        ls.setThickness(w)
        ls.setColor(*color, 1)
        ls.moveTo(Point3(*a))
        ls.drawTo(Point3(*b))
        n_ = base.render.attachNewNode(ls.create())
        n_.setLightOff()
        n_.setDepthTest(False)
        n_.setBin("fixed", 100)
        live.append(n_)

    def dot(x, color, r=0.007):
        T = np.eye(4)
        T[:3, 3] = x
        m = trimesh.creation.icosphere(subdivisions=2, radius=r)
        add(m, T, color)
        live[-1].setDepthTest(False)
        live[-1].setBin("fixed", 101)
        live[-1].setLightOff()

    def tm(mesh):
        return trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=False)

    def table_and_robot():
        add(box(tc - ts / 2, tc + ts / 2), np.eye(4), TABLE)
        top = tc[2] + ts[2] / 2
        for x in np.arange(tc[0] - ts[0] / 2 + 0.05, tc[0] + ts[0] / 2 - 1e-6, 0.05):
            add(box((x - 0.0012, tc[1] - ts[1] / 2, top), (x + 0.0012, tc[1] + ts[1] / 2, top + 0.0004)), np.eye(4), GRID)
        for y in np.arange(tc[1] - ts[1] / 2 + 0.05, tc[1] + ts[1] / 2 - 1e-6, 0.05):
            add(box((tc[0] - ts[0] / 2, y - 0.0012, top), (tc[0] + ts[0] / 2, y + 0.0012, top + 0.0004)), np.eye(4), GRID)
        for _, _, lnk in links:
            add(tm(lnk.cmodel.trm_mesh), lnk.gl_homomat, ROBOT, alpha=0.45)
        add(tm(sc.meshes["base"]), sc.goal["base"], COLORS["base"])

    def clear():
        for n_ in live:
            n_.removeNode()
        live.clear()

    def shot(f):
        look = 0.5 * (sc.goal["base"][:3, 3] + tc) + np.array([-0.01, 0.0, 0.05])
        base.cam.setPos(Point3(*look) + Vec3(0.90, -1.18, 0.95))
        base.cam.lookAt(Point3(*look), Vec3(0, 0, 1))
        for _ in range(4):
            base.graphicsEngine.renderFrame()
        base.graphicsEngine.syncFrame()
        img = PNMImage()
        base.win.getScreenshot(img)
        img.write(Filename.fromOsSpecific(str(f.resolve())))
        return f

    out = OUT / "render"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    try:
        for i, (S, p, c) in enumerate(steps):
            clear()
            table_and_robot()
            for q in S:
                add(tm(sc.meshes[q]), sc.goal[q], COLORS[q])
            add(tm(sc.meshes[p]), sc.goal[p], COLORS[p], alpha=0.35)
            Tx = sc.pose[p][c["pose"]]
            add(tm(sc.meshes[p]), Tx, COLORS[p])
            pick = Tx[:3, 3]
            frm = home if c["last"] is None else goal[c["last"]] + up
            line(frm, pick + up, TRANSIT)
            line(pick + up, pick, VERT, 3.5)
            line(pick + up, goal[p] + up, TRANSFER)
            line(goal[p] + up, goal[p], VERT, 3.5)
            dot(frm, TRANSIT)
            dot(pick + up, (0.3, 0.3, 0.3), 0.005)
            dot(goal[p] + up, TRANSFER, 0.005)
            files.append(shot(out / f"bound_step_{i}.png"))
        clear()                                   # the relaxed layout: every part at its own cheapest pose
        table_and_robot()
        for S, p, c in steps[:6]:
            add(tm(sc.meshes[p]), sc.pose[p][c["pose"]], COLORS[p])
            add(tm(sc.meshes[p]), sc.goal[p], COLORS[p], alpha=0.25)
        files.append(shot(out / "bound_layout.png"))
    finally:
        base.destroy()
    over = []
    for a in range(6):
        for b in range(a + 1, 6):
            (_, p, c), (_, q, e) = steps[a], steps[b]
            if sc.polygons[p][c["pose"]].distance(sc.polygons[q][e["pose"]]) < pb.gap:
                over.append((SHORT[p], SHORT[q]))
    log(f"relaxed layout: footprints closer than {pb.gap} m: {over}")
    compose(steps, files, over)


def compose(steps, files, over):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from .viz_lattice import INK, INK2, SURF
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    hexc = lambda c: "#%02x%02x%02x" % tuple(int(255 * v) for v in c)
    nm = lambda S: "{" + ",".join(sorted((SHORT[q] for q in S), key=["A", "B", "C", "AB", "BC", "CA"].index)) + "}"
    fig = plt.figure(figsize=(20, 17.6), facecolor=SURF)
    W, H0 = 0.325, 0.255
    cells = [(0.01 + j * 0.33, 0.615 - i * 0.30) for i in range(3) for j in range(3)]
    for k, (S, p, c) in enumerate(steps):
        x0, y0 = cells[k]
        ax = fig.add_axes([x0, y0, W, H0])
        ax.imshow(mpimg.imread(files[k])[70:-60, 60:-60])
        ax.axis("off")
        frm = "HOME" if c["last"] is None else f"{SHORT[c['last']]} 的目标上方"
        head = f"{'①②③④⑤⑥'[k]} 箭头 {nm(S)} —装 {SHORT[p]}→" if k < 6 else f"对照：被剪掉的箭头 {nm(S)} —装 {SHORT[p]}→"
        ax.set_title(f"{head}（上一个 = {SHORT.get(c['last'], '无')}，|D| = {c['n_dom']}）\n"
                     f"step = 空手 {c['transit']:.4f}（从 {frm}）+ 抓 0.10 + 搬 {c['transfer']:.4f} + 插 0.10 = {c['step']:.4f}\n"
                     f"F {c['F']:.4f} + step {c['step']:.4f} + H {c['H']:.4f} = {c['total']:.4f}"
                     + ("  ≥ U，剪掉" if c["total"] >= U else "  < U"),
                     fontsize=10.5, color="#c0392b" if c["total"] >= U else INK, loc="left")
    x0, y0 = cells[7]
    ax = fig.add_axes([x0, y0, W, H0])
    ax.imshow(mpimg.imread(files[7])[70:-60, 60:-60])
    ax.axis("off")
    ax.set_title("乐观布局：6 个零件各自摆在自己最省的位置（实心）\n"
                 f"下界不管它们互相挡：{'、'.join(f'{a} 和 {b}' for a, b in over) or '无'} 的位置挨得太近/重叠\n"
                 f"真实方案要让开 → 真实最优 U = {U}，比乐观值 3.0138 大", fontsize=10.5, color=INK, loc="left")
    x0, y0 = cells[8]
    ax = fig.add_axes([x0 + 0.04, y0 + 0.02, W - 0.05, H0 - 0.03])
    labels = [f"{'①②③④⑤⑥'[k]} →{SHORT[p]}" if k < 6 else f"对照 →{SHORT[p]}" for k, (S, p, c) in enumerate(steps)]
    for k, (S, p, c) in enumerate(steps):
        y = len(steps) - 1 - k
        ax.barh(y, c["F"], color="#c9c7c1", height=0.6)
        ax.barh(y, c["step"], left=c["F"], color=hexc(COLORS[p]), height=0.6)
        ax.barh(y, c["H"], left=c["F"] + c["step"], color="#e8e6e1", height=0.6, edgecolor="#c9c7c1", lw=0.6)
        ax.text(3.42, y, f"{c['total']:.4f}", va="center", fontsize=9.5,
                color="#c0392b" if c["total"] >= U else INK)
    ax.axvline(U, color="#c0392b", lw=1.2, ls="--")
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels(labels[::-1], fontsize=9.5)
    ax.set_xlim(0, 3.85)
    ax.set_ylim(-0.6, len(steps) - 0.4)
    ax.set_xlabel("代价（m）", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"每条箭头 = F（灰：前面已花）+ step（彩：这一步）+ H（浅：后面还要花）\n红虚线 = 真实最优 U = {U}", fontsize=10.5, color=INK, loc="left")
    fig.text(0.01, 0.975, "箭头上的乐观值是怎么算的（chain 代价）：沿乐观最优顺序 A→B→AB→C→BC→CA 逐步看", fontsize=15, color=INK)
    fig.text(0.01, 0.958, "每张图：已装好的零件 S（实色在目标处），这一步要装的零件 p 在 D(S, p) 里最省的桌面位置（实色在桌上）"
                          "和它的目标（半透明）。蓝线 = 空手移过去，品红线 = 拿着搬过去，黑竖线 = 抓取 / 插入（各记 0.10 m）",
             fontsize=11, color=INK2)
    fig.text(0.01, 0.943, "F = 从 HOME 到这一步之前的乐观代价（从前往后累加）；H = 装完这一步后剩下的乐观代价（含最后回 HOME）。"
                          "最优路径上每条箭头的 F + step + H 都等于 3.0138", fontsize=11, color=INK2)
    f = OUT / "bound_steps_triangular_frame6.png"
    fig.savefig(f, dpi=115, facecolor=SURF)
    log(f"-> {f}")


if __name__ == "__main__":
    sys.exit(main())
