"""Step 1 of the lattice search: the table poses D(S, p) of part p usable under its installed set S (2026-11-01).

For every arrow (S --p--> S + {p}) of the frame's state graph, with the checks of run.py /
lazy.installed_checks (sphere model, lazy_cost.checker.GpuChecker), options = (pose x, grasp g), k
grasps per pose (build_options):
  pick side  ok[x, g]: hover / pick / lift (held) / descent / pick -> lift against the table, the
             assembly base and the installed set S
  goal side  okg[g]  : preins (held) / ins / preins -> ins / release line against S, retreat against S + {p}
  D(S, p) = poses x with some grasp g: ok[x, g] and okg[g]  (the same grasp picks p and inserts it)
A position (x, y) is in D when one of its orientations is (the mainline uses one orientation per position).
compute: logs every arrow and saves _out/domain_triangular_frame6.json.
render : Panda3D pictures of chosen arrows: installed set S in colour, p at its goal (ghost), p's
         candidate poses on the table, green = in D(S, p), red = not; composed with matplotlib.

    python -X utf8 -m experiments.bls_all_orders_20261031.viz_domain compute
    python -X utf8 -m experiments.bls_all_orders_20261031.viz_domain render
"""
from __future__ import annotations

import argparse
import json
import sys

from curobo_bridge.runtime import prepare

prepare()

import numpy as np  # noqa: E402

from .run import OUT, build_options, installed_checks, log, stable_orders  # noqa: E402

MODEL, GOAL_POS, K = "triangular_frame6", "0.373,0.0,0.0", 8
SHORT = {"post_A": "A", "post_B": "B", "post_C": "C", "beam_AB": "AB", "beam_BC": "BC", "beam_CA": "CA"}
JSON = OUT / f"domain_{MODEL}.json"


def compute():
    from joint_search.problem import Problem
    from lazy_cost.scene import key_xy
    pb = Problem(MODEL, center=tuple(float(v) for v in GOAL_POS.split(",")))
    orders = stable_orders(pb)
    O = build_options(pb, K)
    edges = sorted({(tuple(sorted(o[:k])), o[k]) for o in orders for k in range(len(o))}, key=lambda e: (len(e[0]), e))
    bp, bg = installed_checks(pb, O, edges)
    out = []
    for S, p in edges:
        poses, opt = O["poses"][p], O["opt"][p]
        ok, okg = bp[S, p], bg[S, p]
        xi = np.array([x for x, _ in opt])
        gi = np.array([g for _, g in opt])
        pick_any = np.zeros(len(poses), bool)
        np.logical_or.at(pick_any, xi, ok)
        dom = np.zeros(len(poses), bool)
        np.logical_or.at(dom, xi, ok & okg[gi])
        pos = {}
        for i, x in enumerate(poses):                       # per position: in D / why not
            st = "D" if dom[i] else ("goal" if pick_any[i] else "pick")
            k = key_xy(x)
            prev = pos.get(k)
            if prev is None or "D goal pick".index(st) < "D goal pick".index(prev[1]):
                pos[k] = (x, st)
        rec = dict(S=list(S), p=p, n_grasp=len(O["gl"][p]), n_goal_ok=int(okg.sum()),
                   goal_ok=[O["gl"][p][g] for g in np.nonzero(okg)[0]],
                   n_opt=len(opt), n_opt_pick_ok=int(ok.sum()), n_opt_ok=int((ok & okg[gi]).sum()),
                   positions={k: v for k, v in pos.items()})
        out.append(rec)
        c = [s for _, s in pos.values()]
        log(f"  {'+'.join(SHORT[q] for q in S) or '{}':16s} --{SHORT[p]:2s}->  goal grasps ok {rec['n_goal_ok']:2d}/{rec['n_grasp']:2d}"
            f"   options pick-ok {rec['n_opt_pick_ok']:3d}/{rec['n_opt']:3d}, both {rec['n_opt_ok']:3d}"
            f"   positions in D {c.count('D'):2d}/{len(c)}  (pick blocked {c.count('pick')}, all grasps blocked at goal {c.count('goal')})")
    sc = pb.scene
    JSON.write_text(json.dumps(dict(center=list(map(float, sc.center)) if getattr(sc, "center", None) is not None else None,
                                    edges=out, orders=[list(o) for o in orders]), indent=1), encoding="utf-8")
    log(f"-> {JSON}")


COLORS = {"post_A": (0.20, 0.56, 0.27), "post_B": (0.80, 0.24, 0.24), "post_C": (0.23, 0.35, 0.58),
          "beam_AB": (0.93, 0.56, 0.18), "beam_BC": (0.90, 0.76, 0.22), "beam_CA": (0.62, 0.36, 0.66),
          "base": (0.47, 0.58, 0.72)}
IN_D, NOT_D = (0.16, 0.62, 0.30), (0.85, 0.20, 0.18)
PART = "beam_BC"
ARROWS = [("post_B", "post_C"), ("beam_CA", "post_A", "post_B", "post_C"), ("beam_AB", "beam_CA", "post_A", "post_B", "post_C")]


def render():
    import trimesh
    import wrs.basis.trimesh as wtrm
    import wrs.modeling.collision_model as mcm
    from panda3d.core import (AmbientLight, DirectionalLight, Filename, PNMImage, Point3, TransparencyAttrib, Vec3,
                              Vec4, loadPrcFileData)
    from adapters.panthera import PantheraArmProvider
    from curobo_bridge.export_model import wrs_collision_parts
    from joint_search.problem import Problem
    from lazy_cost.scene import FRAME_DIR
    from .render_frame import GRID, ROBOT, TABLE, box

    d = json.loads(JSON.read_text(encoding="utf-8"))
    E = {(tuple(sorted(e["S"])), e["p"]): e for e in d["edges"]}
    pb = Problem(MODEL, center=tuple(float(v) for v in GOAL_POS.split(",")))
    sc, cat = pb.scene, pb.cat
    tb = json.loads((FRAME_DIR / "task.json").read_text(encoding="utf-8"))["table"]
    tc, ts = np.asarray(sc.table_center, float), np.asarray(sc.table_size, float)
    arm = PantheraArmProvider(base_pos=tb["robot_base_pos"], base_rotmat=tb["robot_base_rotmat"]).arm()
    links = wrs_collision_parts(arm, include_base=True)

    def cm(mesh, T):
        c = mcm.CollisionModel(initor=wtrm.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces)))
        c.pose = (T[:3, 3].copy(), T[:3, :3].copy())
        return c
    cms = {q: cm(sc.meshes[q], sc.goal[q]) for q in sc.ids}
    # the grasp shown in the close-ups: usable with {B, C}, blocked with the largest S, and its arm at the
    # insertion (or above it) visibly hits a part of that S (WRS meshes); plus one grasp usable everywhere
    keys = [tuple(sorted(S)) for S in ARROWS]
    ok = [set(E[S, PART]["goal_ok"]) for S in keys]
    extra = [cms[q] for q in keys[-1] if q not in keys[0]]
    shown_bad = None
    for g in sorted(ok[0] - ok[-1]):
        for tag in ("ins", "preins"):
            arm.goto_given_conf(np.asarray(cat["goal"][PART][g][tag], float))
            jw, jr = float(pb.jaw[PART][g]), arm.end_effector.jaw_range
            arm.change_jaw_width(float(jr[1] if not np.isfinite(jw) else np.clip(jw, *jr)))
            hit = [lnk for _, _, lnk in links[1:] if lnk.cmodel.is_mcdwith(extra)]
            if hit:
                shown_bad = (g, tag)
                break
        if shown_bad:
            break
    shown_ok = sorted(set.intersection(*ok))[0]
    log(f"close-ups: grasp {shown_bad} ok with {{B,C}}, blocked later; grasp {shown_ok} ok in all three")

    loadPrcFileData("", "window-type offscreen\nwin-size 1500 1050\naudio-library-name null\n"
                        "framebuffer-multisample 1\nmultisamples 8")
    from direct.showbase.ShowBase import ShowBase
    from wrs.basis.data_adapter import npmat4_to_pdmat4, pdgeomndp_from_vvnf
    base = ShowBase(windowType="offscreen")
    base.disableMouse()
    base.setBackgroundColor(1, 1, 1, 1)
    base.camLens.setNearFar(0.01, 20.0)
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
        m = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=False)
        node = pdgeomndp_from_vvnf(np.asarray(m.vertices, np.float32), np.asarray(m.vertex_normals, np.float32),
                                   np.asarray(m.faces, np.uint32), f"n{len(live)}")
        node.reparentTo(base.render)
        node.setMat(npmat4_to_pdmat4(np.asarray(pose, float)))
        node.setColor(*color, alpha)
        node.setTwoSided(True)
        if alpha < 1:
            node.setTransparency(TransparencyAttrib.MAlpha)
            node.setDepthWrite(False)
            node.setBin("fixed", 50)
        live.append(node)

    def scene(S, cands, q=None, jaw=None, arm_color=ROBOT):
        for n_ in live:
            n_.removeNode()
        live.clear()
        add(box(tc - ts / 2, tc + ts / 2), np.eye(4), TABLE)
        top = tc[2] + ts[2] / 2
        for x in np.arange(tc[0] - ts[0] / 2 + 0.05, tc[0] + ts[0] / 2 - 1e-6, 0.05):
            add(box((x - 0.0012, tc[1] - ts[1] / 2, top), (x + 0.0012, tc[1] + ts[1] / 2, top + 0.0004)), np.eye(4), GRID)
        for y in np.arange(tc[1] - ts[1] / 2 + 0.05, tc[1] + ts[1] / 2 - 1e-6, 0.05):
            add(box((tc[0] - ts[0] / 2, y - 0.0012, top), (tc[0] + ts[0] / 2, y + 0.0012, top + 0.0004)), np.eye(4), GRID)
        arm.goto_given_conf(np.zeros(6) if q is None else np.asarray(q, float))
        jr = arm.end_effector.jaw_range
        arm.change_jaw_width(float(jr[1] if jaw is None or not np.isfinite(jaw) else np.clip(jaw, *jr)))
        for _, _, lnk in links:
            add(lnk.cmodel.trm_mesh, lnk.gl_homomat, arm_color)
        add(sc.meshes["base"], sc.goal["base"], COLORS["base"])
        for qid in S:
            add(sc.meshes[qid], sc.goal[qid], COLORS[qid])
        add(sc.meshes[PART], sc.goal[PART], COLORS[PART], alpha=0.35 if q is None else 1.0)
        for key, st in cands:
            add(sc.meshes[PART], sc.pose[PART][key], IN_D if st == "D" else NOT_D)

    def shot(f, look, off, fov):
        base.camLens.setFov(fov)
        base.cam.setPos(Point3(*look) + Vec3(*off))
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
    site = sc.goal["base"][:3, 3]
    wide_look = 0.5 * (site + tc) + np.array([-0.02, 0.0, 0.04])
    goal_look = sc.goal[PART][:3, 3] + np.array([0.0, 0.0, 0.03])
    files = []
    try:
        for S in keys:
            e = E[S, PART]
            cands = list(e["positions"].values())
            tag = "_".join(SHORT[q] for q in S)
            scene(S, cands)
            fw = shot(out / f"domain_{SHORT[PART]}_{tag}_wide.png", wide_look, (0.95, -1.30, 0.98), 32)
            g, cfg = shown_bad
            good = g in e["goal_ok"]
            scene(S, [], q=cat["goal"][PART][g][cfg], jaw=pb.jaw[PART][g],
                  arm_color=(0.45, 0.70, 0.48) if good else (0.88, 0.42, 0.38))
            fc = shot(out / f"domain_{SHORT[PART]}_{tag}_grasp.png", goal_look, (0.42, -0.55, 0.40), 34)
            files.append((S, e, fw, fc, good))
    finally:
        base.destroy()
    compose(files, shown_bad)


def compose(files, shown_bad):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from .viz_lattice import INK, INK2, SURF
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(19, 11.6), facecolor=SURF)
    n = len(files)
    for i, (S, e, fw, fc, good) in enumerate(files):
        c = [st for _, st in e["positions"].values()]
        name = "{" + ",".join(SHORT[q] for q in S) + "}"
        ax = fig.add_axes([0.01 + i / n * 0.98, 0.53, 0.98 / n - 0.01, 0.34])
        ax.imshow(mpimg.imread(fw)[60:-40, 40:-40])
        ax.axis("off")
        ax.set_title(f"箭头 {name} —装 BC→\nD(S, BC) = {c.count('D')} / {len(c)} 个桌面位置"
                     f"（红 {c.count('goal')} 个：它们的抓取在目标处全被挡）", fontsize=11.5, color=INK, loc="left")
        ax2 = fig.add_axes([0.01 + i / n * 0.98, 0.06, 0.98 / n - 0.01, 0.39])
        ax2.imshow(mpimg.imread(fc)[80:-60, 120:-120])
        ax2.axis("off")
        ax2.set_title(f"目标处能用的抓取 {e['n_goal_ok']} / {e['n_grasp']}；图中抓取 #{shown_bad[0]}："
                      + ("能插入" if good else "手臂撞到已装零件，不能用"), fontsize=10.5,
                      color="#2f8f45" if good else "#c0392b", loc="left")
    fig.text(0.01, 0.965, "第 1 步：每条箭头算一次 D(S, p)。以 beam_BC（黄色横梁，半透明 = 它的目标位置）为例，"
                          "三条装 BC 的箭头只差在“已经装了哪些零件 S”", fontsize=14, color=INK)
    fig.text(0.01, 0.935, "上排：桌上每个方块 = BC 的一个候选摆放位置（共 40 个）。绿 = 在 D(S, BC) 里，红 = 被删掉。"
                          "有的方块竖着，是 SEALP 候选朝向里的立放。下排：同一个抓取 #91 在 BC 目标处插入时的手臂姿态（IK 目录里的解）", fontsize=11, color=INK2)
    fig.text(0.01, 0.02, "怎么算：一个桌面位置 x 留在 D 里，当且仅当存在一个抓取 g，(1) 机械臂在 x 处用 g 抓起 BC 不撞桌子、底座和 S；"
                         "(2) 用同一个 g 在目标处插入、退出时不撞底座和 S。"
                         "S 越大，(2) 里被挡的抓取越多；x 的几个抓取全被挡，x 就变红。(1) 在三角框架上从不被 S 挡（桌面离装配区远）",
             fontsize=10, color=INK2)
    f = OUT / f"domain_{SHORT[PART]}_triangular_frame6.png"
    fig.savefig(f, dpi=120, facecolor=SURF)
    log(f"-> {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("compute", "render"))
    a = ap.parse_args()
    sys.exit(compute() if a.what == "compute" else render())
