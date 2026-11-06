"""Full arm + every scene part, as each collision checker sees them (same configuration).

    & $py -X utf8 -m joint_search.viz_collision_full

Scene: triangular frame mid-assembly (posts + beam_AB installed, beam_CA on the table); the arm
holds beam_BC 5 cm above its goal (gripper closed to the grasp's jaw).
  1. cuRobo (our GPU check)   robot = spheres (+ buffer), fingers closed to the jaw; held part = its
                              surface sphere cover (+ buffer); installed parts, parts on the table and
                              the assembly base = their triangle meshes; table = a cuboid
  2. WRS mesh check            every object = its triangle mesh (the reference of our experiments)
  3. WRS RRT internal check    every object = its axis-aligned box in its own frame + 2 mm
                              (CDPrimType.AABB, used by WRS RRT-Connect while planning)
Output: joint_search/_output/viz/collision_full.png, collision_full.html (interactive, 3 modes).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .viz_collision_models import LINK_COL, simplify, spheres_to_mesh

OUT = Path(__file__).resolve().parent / "_output" / "viz"
BOX_F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])


def box_world(V_local, pos, rotmat, pad):
    lo, hi = V_local.min(0) - pad, V_local.max(0) + pad
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return c @ np.asarray(rotmat).T + np.asarray(pos), BOX_F


def main():
    import torch
    from .problem import Problem
    from .wrs_check import MeshWorld
    from .exp_collision_bench import place_near
    OUT.mkdir(parents=True, exist_ok=True)
    pb = Problem("triangular_frame6")
    sc, cat, chk = pb.scene, pb.cat, pb.chk
    W = MeshWorld(pb)
    p, installed = "beam_BC", ["post_A", "post_B", "post_C", "beam_AB"]
    table = place_near(sc, ["beam_CA"], cat, 0.01)
    pairs = sorted(((x, g) for x in cat["pick"][p] for g in cat["pick"][p][x]),
                   key=lambda xg: abs(sc.pose[p][xg[0]][1, 3]) + abs(sc.pose[p][xg[0]][0, 3] - 0.25))
    x, g = pairs[0]
    jaw, G = pb.jaw[p][g], sc.grasps()[p][g][0]
    q = np.asarray(cat["goal"][p][g]["preins"], float)

    # ---- scene objects (pose, mesh) --------------------------------------------------------------------
    objs = [("base", sc.goal["base"], "#9e9e9e")] + [(o, sc.goal[o], "#4c78a8") for o in installed] + \
           [(o, sc.pose[o][k], "#f58518") for o, k in table.items()]
    world_mesh = lambda o, T: (np.asarray(sc.meshes[o].vertices) @ T[:3, :3].T + T[:3, 3], np.asarray(sc.meshes[o].faces))
    tc, ts = np.asarray(sc.table_center, float), np.asarray(sc.table_size, float)
    table_box = box_world(np.array([-ts / 2, ts / 2]), tc, np.eye(3), 0.0)

    # ---- 1. cuRobo -------------------------------------------------------------------------------------
    sph, R, pt = chk.kinematics(torch.as_tensor(q[None], dtype=torch.float32, device="cuda"))
    sph = sph.clone()
    chk._close_fingers(sph, R, torch.tensor([jaw], device="cuda", dtype=torch.float32))
    robot_s = sph[0].cpu().numpy().astype(float)
    robot_s[:, 3] += chk.buffer
    kc = chk.chk.kinematics.kinematics_config
    names = {v: k for k, v in kc.link_name_to_idx_map.items()}
    link_of = [names[int(i)] for i in kc.link_sphere_idx_map.cpu().numpy()]
    held_s = chk.held_world(R, pt, pb.held[p], G)[0].cpu().numpy().astype(float)
    tcp = np.eye(4)
    tcp[:3, :3], tcp[:3, 3] = R[0].cpu().numpy(), pt[0].cpu().numpy()
    cu = []
    for ln in sorted(set(link_of)):
        S = robot_s[[i for i, l in enumerate(link_of) if l == ln]]
        cu.append((f"robot spheres: {ln} ({len(S)})", *spheres_to_mesh(S), LINK_COL.get(ln, "#7f7f7f"), 0.55))
    cu.append((f"held part: {len(held_s)} spheres", *spheres_to_mesh(held_s), "#54a24b", 0.6))
    for o, T, col in objs:
        cu.append((f"{o}: mesh", *simplify(*world_mesh(o, T)), col, 0.45))
    cu.append(("table: cuboid", *table_box, "#d9d9d9", 0.25))

    # ---- 2. WRS mesh / 3. WRS AABB primitives ---------------------------------------------------------
    W.set(installed, {o: sc.pose[o][k] for o, k in table.items()}, (p, G, q), jaw=jaw)
    W.arm.goto_given_conf(q)
    W.place_held()
    wm, wb, types = [], [], {}
    for ln, l in W.parts.items():
        cm = l.cmodel
        V0, F0 = np.asarray(cm.trm_mesh.vertices), np.asarray(cm.trm_mesh.faces)
        col = LINK_COL.get(ln, "#7f7f7f")
        wm.append((f"robot mesh: {ln} ({len(F0)} triangles)", *simplify(V0 @ cm.rotmat.T + cm.pos, F0), col, 0.8))
        types[ln] = str(cm.cdprim_type)
        wb.append((f"robot AABB: {ln}", *box_world(V0, cm.pos, cm.rotmat, 0.002), col, 0.35))
    hc = W.held
    Vh, Fh = np.asarray(hc.trm_mesh.vertices), np.asarray(hc.trm_mesh.faces)
    wm.append(("held part: mesh", Vh @ hc.rotmat.T + hc.pos, Fh, "#54a24b", 0.85))
    wb.append(("held part: AABB", *box_world(Vh, hc.pos, hc.rotmat, 0.002), "#54a24b", 0.45))
    for o, T, col in objs:
        V, F = world_mesh(o, T)
        wm.append((f"{o}: mesh", *simplify(V, F), col, 0.45))
        wb.append((f"{o}: AABB", *box_world(np.asarray(sc.meshes[o].vertices), T[:3, 3], T[:3, :3], 0.002), col, 0.3))
    wm.append(("table: mesh (box)", *table_box, "#d9d9d9", 0.25))
    wb.append(("table: AABB", *box_world(np.array([-ts / 2, ts / 2]), tc, np.eye(3), 0.002), "#d9d9d9", 0.25))
    free_gpu = bool(chk.batch_free([(q[None], None, 0, -1, np.array([jaw]))], held_bank=[(pb.held[p], G)])[0][0])
    free_wrs = bool(W.free([q, q])[0])
    free_aabb = not bool(W.arm.is_collided(obstacle_list=W.obs))
    W.set([], {})
    print("cdprim types of the robot links:", types)
    print(f"verdict: cuRobo {'free' if free_gpu else 'COLLISION'}, WRS mesh {'free' if free_wrs else 'COLLISION'}, "
          f"WRS AABB {'free' if free_aabb else 'COLLISION'}")
    cols = [("1. cuRobo (GPU): robot = spheres, held part = sphere cover, scene = meshes", cu, free_gpu),
            ("2. WRS mesh check: everything = triangle meshes", wm, free_wrs),
            ("3. WRS RRT internal: everything = axis-aligned box in its own frame + 2 mm", wb, free_aabb)]
    meta = dict(part=p, grasp=int(g), jaw=jaw, installed=installed, table=table, density=chk.density, buffer_mm=1e3 * chk.buffer,
                cdprim_types=types)
    render_png(cols, tcp, meta)
    render_html(cols, tcp, meta)


def render_png(cols, tcp, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(21, 14))
    c = tcp[:3, 3]
    for row, (lim, view) in enumerate(((((-0.12, 0.58), (-0.35, 0.35), (-0.02, 0.62)), (20, -60)),
                                       (((c[0] - 0.11, c[0] + 0.11), (c[1] - 0.11, c[1] + 0.11), (max(c[2] - 0.14, -0.01), c[2] + 0.08)), (18, -50)))):
        for k, (title, items, free) in enumerate(cols):
            ax = fig.add_subplot(2, 3, 3 * row + k + 1, projection="3d")
            for name, V, F, col, op in items:
                ax.add_collection3d(Poly3DCollection(V[F], facecolor=col, edgecolor="k" if "AABB" in name else "none",
                                                     linewidths=0.3 if "AABB" in name else 0, alpha=op))
            ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
            ax.set_box_aspect([l[1] - l[0] for l in lim]); ax.view_init(*view)
            ax.set_title(("" if row else title + "\n") + ("full view" if row == 0 else "zoom: gripper, held part, posts")
                         + f" - verdict: {'free' if free else 'COLLISION'}", fontsize=10)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"Same configuration in three collision models - holding {meta['part']} (grasp {meta['grasp']}, jaw {1e3 * meta['jaw']:.0f} mm) "
                 f"5 cm above its goal. Blue = installed, orange = on the table, grey = base / table, green = held part, red = fingers, "
                 f"purple = palm. cuRobo spheres: density {meta['density']:g} + {meta['buffer_mm']:.0f} mm.", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "collision_full.png", dpi=105)
    plt.close(fig)
    print("->", OUT / "collision_full.png")


def render_html(cols, tcp, meta):
    figs = []
    for title, items, free in cols:
        figs.append(dict(title=title, free=free, traces=[dict(
            type="mesh3d", x=V[:, 0].round(4).tolist(), y=V[:, 1].round(4).tolist(), z=V[:, 2].round(4).tolist(),
            i=F[:, 0].tolist(), j=F[:, 1].tolist(), k=F[:, 2].tolist(), color=col, opacity=op, name=name,
            flatshading=True, showlegend=True) for name, V, F, col, op in items]))
    page = """<!doctype html><html><head><meta charset="utf-8"><title>Collision models: full view</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>body{font-family:sans-serif;margin:10px}button{margin:2px 4px;padding:5px 12px;font-size:14px}#plot{width:100%;height:85vh}</style>
</head><body><div id="btns"></div><div id="plot"></div><script>
const FIGS=__DATA__; let cur=0;
function draw(){const f=FIGS[cur];
 Plotly.react('plot',f.traces,{margin:{l:0,r:0,t:40,b:0},title:f.title+'  -  verdict: '+(f.free?'free':'COLLISION'),
  legend:{font:{size:11}},scene:{aspectmode:'data',camera:{eye:{x:-1.2,y:-1.6,z:0.9}}}});}
document.getElementById('btns').innerHTML=FIGS.map((f,i)=>`<button onclick="cur=${i};draw()">${f.title.split(':')[0]}</button>`).join('');
draw();</script></body></html>"""
    (OUT / "collision_full.html").write_text(page.replace("__DATA__", json.dumps(figs)), encoding="utf-8")
    print("->", OUT / "collision_full.html")


if __name__ == "__main__":
    main()
