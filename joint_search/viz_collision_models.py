"""Visualise the two collision models on the same scene and configuration.

    & $py -X utf8 -m joint_search.viz_collision_models

Scene: triangular frame mid-assembly (posts + beam_AB installed, beam_CA on the table), the arm
handles beam_BC. Two configurations of one grasp:
  1. above insert   holding beam_BC 5 cm above its goal, gripper closed, next to the posts
  2. pick moment    gripper closed on beam_BC lying on the table
Left: what our GPU check (cuRobo) sees - robot spheres (+ buffer) with the fingers closed to the
grasp's jaw, the held part as its surface sphere cover (+ buffer); obstacles are their meshes.
Right: what WRS sees - the triangle meshes of the robot links, the gripper and the held part.
Output: joint_search/_output/viz/collision_models.png and collision_models.html (plotly, interactive).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "_output" / "viz"


def icosphere():
    import trimesh
    s = trimesh.creation.icosphere(subdivisions=1)
    return np.asarray(s.vertices), np.asarray(s.faces)


def simplify(V, F, target=1500):
    """Quadric decimation (open3d) for display only: the WRS link meshes have ~10^4-10^5 triangles."""
    if len(F) <= target:
        return V, F
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    m = m.simplify_quadric_decimation(target_number_of_triangles=target)
    return np.asarray(m.vertices), np.asarray(m.triangles)


def spheres_to_mesh(S):
    """S [n,4] (x, y, z, r) -> one triangle mesh (vertices, faces)."""
    v0, f0 = icosphere()
    V = (S[:, None, 3:4] * v0[None] + S[:, None, :3]).reshape(-1, 3)
    F = (f0[None] + (np.arange(len(S)) * len(v0))[:, None, None]).reshape(-1, 3)
    return V, F


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
    # a grasp / pose whose two configurations both exist; prefer a pose in front of the robot
    pairs = sorted(((x, g) for x in cat["pick"][p] for g in cat["pick"][p][x]),
                   key=lambda xg: abs(sc.pose[p][xg[0]][1, 3]) + abs(sc.pose[p][xg[0]][0, 3] - 0.25))
    x, g = pairs[0]
    jaw = pb.jaw[p][g]
    G = sc.grasps()[p][g][0]
    configs = {"1_above_insert_holding": (cat["goal"][p][g]["preins"], True, False),
               "2_pick_moment": (cat["pick"][p][x][g]["pick"], False, True)}
    scene_meshes = [("base", sc.goal["base"], "#9e9e9e")] + [(q, sc.goal[q], "#4c78a8") for q in installed] + \
                   [(q, sc.pose[q][k], "#f58518") for q, k in table.items()]
    data = {}
    for name, (q, holding, part_on_table) in configs.items():
        # ---- cuRobo view -------------------------------------------------------------------------
        sph, R, pt = chk.kinematics(torch.as_tensor(np.asarray(q)[None], dtype=torch.float32, device="cuda"))
        sph = sph.clone()
        chk._close_fingers(sph, R, torch.tensor([jaw], device="cuda", dtype=torch.float32))
        robot_s = sph[0].cpu().numpy().astype(float)
        robot_s[:, 3] += chk.buffer
        kc = chk.chk.kinematics.kinematics_config
        names = {v: k for k, v in kc.link_name_to_idx_map.items()}
        link_of = [names[int(i)] for i in kc.link_sphere_idx_map.cpu().numpy()]
        held_s = chk.held_world(R, pt, pb.held[p], G)[0].cpu().numpy().astype(float) if holding else np.zeros((0, 4))
        tcp = np.eye(4)
        tcp[:3, :3], tcp[:3, 3] = R[0].cpu().numpy(), pt[0].cpu().numpy()
        # ---- WRS view ----------------------------------------------------------------------------
        W.set(installed, {q_: sc.pose[q_][k] for q_, k in table.items()},
              (p, G, q) if holding else None, jaw=jaw)
        W.arm.goto_given_conf(np.asarray(q, float))
        W.place_held()
        links = []
        for ln, l in W.parts.items():
            cm = l.cmodel
            V = np.asarray(cm.trm_mesh.vertices) @ cm.rotmat.T + cm.pos
            links.append((ln, V, np.asarray(cm.trm_mesh.faces)))
        held_mesh = None
        if holding:
            T = tcp @ np.linalg.inv(G)
            m = sc.meshes[p]
            held_mesh = (np.asarray(m.vertices) @ T[:3, :3].T + T[:3, 3], np.asarray(m.faces))
        part_table = None
        if part_on_table:
            T = sc.pose[p][x]
            part_table = (np.asarray(sc.meshes[p].vertices) @ T[:3, :3].T + T[:3, 3], np.asarray(sc.meshes[p].faces))
        free_gpu = bool(chk.batch_free([(np.asarray(q)[None], None, 0 if holding else -1, -1, np.array([jaw]))],
                                       held_bank=[(pb.held[p], G)] if holding else [])[0][0]) if True else None
        free_wrs = bool(W.free([q, q])[0])
        data[name] = dict(robot_s=robot_s, link_of=link_of, held_s=held_s, links=links, held_mesh=held_mesh,
                          part_table=part_table, tcp=tcp, free_gpu=free_gpu, free_wrs=free_wrs)
        print(f"{name}: GPU {'free' if free_gpu else 'COLLISION'}, WRS {'free' if free_wrs else 'COLLISION'}; "
              f"robot spheres {len(robot_s)}, held-part spheres {len(held_s)}", flush=True)
    W.set([], {})
    meta = dict(part=p, pose=x, grasp=int(g), jaw=jaw, installed=installed, table=table,
                robot_buffer_mm=1e3 * chk.buffer, density=chk.density)
    meshes = [(n, np.asarray(sc.meshes[n].vertices) @ T[:3, :3].T + T[:3, 3], np.asarray(sc.meshes[n].faces), c)
              for n, T, c in scene_meshes]
    png(data, meshes, meta)
    html(data, meshes, meta)


LINK_COL = {"finger_left": "#d62728", "finger_right": "#d62728", "gripper_palm": "#9467bd"}


def png(data, meshes, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(16, 13))
    for row, (name, d) in enumerate(data.items()):
        c = d["tcp"][:3, 3]
        for col, view in enumerate(("cuRobo (GPU spheres)", "WRS (triangle meshes)")):
            ax = fig.add_subplot(2, 2, 2 * row + col + 1, projection="3d")
            for n, V, F, colr in meshes:
                ax.add_collection3d(Poly3DCollection(V[F], facecolor=colr, edgecolor="none", alpha=0.35))
            if d["part_table"] is not None:
                V, F = d["part_table"]
                ax.add_collection3d(Poly3DCollection(V[F], facecolor="#54a24b", edgecolor="none", alpha=0.35))
            if col == 0:
                for ln in sorted(set(d["link_of"])):
                    S = d["robot_s"][[i for i, l in enumerate(d["link_of"]) if l == ln]]
                    V, F = spheres_to_mesh(S)
                    ax.add_collection3d(Poly3DCollection(V[F], facecolor=LINK_COL.get(ln, "#7f7f7f"), edgecolor="k",
                                                         linewidths=0.05, alpha=0.55))
                if len(d["held_s"]):
                    V, F = spheres_to_mesh(d["held_s"])
                    ax.add_collection3d(Poly3DCollection(V[F], facecolor="#54a24b", edgecolor="none", alpha=0.5))
            else:
                for ln, V, F in d["links"]:
                    ax.add_collection3d(Poly3DCollection(V[F], facecolor=LINK_COL.get(ln, "#7f7f7f"), edgecolor="k",
                                                         linewidths=0.05, alpha=0.7))
                if d["held_mesh"] is not None:
                    V, F = d["held_mesh"]
                    ax.add_collection3d(Poly3DCollection(V[F], facecolor="#54a24b", edgecolor="k", linewidths=0.2, alpha=0.8))
            r = 0.13
            ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(max(c[2] - r, -0.01), c[2] + r)
            ax.set_box_aspect((1, 1, 1)); ax.view_init(22, -55)
            verdict = d["free_gpu"] if col == 0 else d["free_wrs"]
            ax.set_title(f"{name.split('_', 1)[1].replace('_', ' ')} - {view}\n"
                         f"verdict: {'free' if verdict else 'COLLISION'}", fontsize=11)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.suptitle(f"Same configuration, two collision models - {meta['part']} grasp {meta['grasp']} (jaw {1e3 * meta['jaw']:.0f} mm); "
                 f"robot spheres density {meta['density']:g} + {meta['robot_buffer_mm']:.0f} mm; "
                 "blue = installed, orange = on the table, green = the part being handled, red = fingers", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "collision_models.png", dpi=110)
    plt.close(fig)
    print("->", OUT / "collision_models.png")


def html(data, meshes, meta):
    def mesh(V, F, col, name, op, group, visible=True):
        return dict(type="mesh3d", x=V[:, 0].round(5).tolist(), y=V[:, 1].round(5).tolist(), z=V[:, 2].round(5).tolist(),
                    i=F[:, 0].tolist(), j=F[:, 1].tolist(), k=F[:, 2].tolist(), color=col, opacity=op, name=name,
                    legendgroup=group, showlegend=True, flatshading=True, visible=visible)
    figs = {}
    for name, d in data.items():
        traces = [mesh(*simplify(V, F), c, f"scene: {n}", 0.35, "scene") for n, V, F, c in meshes]
        if d["part_table"] is not None:
            traces.append(mesh(*d["part_table"], "#54a24b", "part on the table (to be picked)", 0.4, "scene"))
        n_scene = len(traces)
        for ln in sorted(set(d["link_of"])):
            S = d["robot_s"][[i for i, l in enumerate(d["link_of"]) if l == ln]]
            traces.append(mesh(*spheres_to_mesh(S), LINK_COL.get(ln, "#7f7f7f"), f"cuRobo spheres: {ln}", 0.55, "curobo"))
        if len(d["held_s"]):
            traces.append(mesh(*spheres_to_mesh(d["held_s"]), "#54a24b", "cuRobo: held part sphere cover", 0.5, "curobo"))
        n_cu = len(traces)
        for ln, V, F in d["links"]:
            traces.append(mesh(*simplify(V, F), LINK_COL.get(ln, "#7f7f7f"), f"WRS mesh: {ln}", 0.8, "wrs", False))
        if d["held_mesh"] is not None:
            traces.append(mesh(*d["held_mesh"], "#54a24b", "WRS: held part mesh", 0.85, "wrs", False))
        c = d["tcp"][:3, 3]
        figs[name] = dict(traces=traces, n_scene=n_scene, n_cu=n_cu, center=c.tolist(),
                          verdict=dict(gpu=d["free_gpu"], wrs=d["free_wrs"]))
    page = """<!doctype html><html><head><meta charset="utf-8"><title>Collision models</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>body{font-family:sans-serif;margin:12px}button{margin:2px 4px;padding:4px 10px}#plot{width:100%;height:82vh}</style></head><body>
<div><b>Configuration:</b> <span id="cfgs"></span> &nbsp; <b>Show:</b>
<button onclick="show('curobo')">cuRobo spheres</button><button onclick="show('wrs')">WRS meshes</button><button onclick="show('both')">both</button>
&nbsp; <span id="verdict"></span></div><div id="plot"></div>
<script>
const FIGS = __DATA__; const META = __META__; let cur = Object.keys(FIGS)[0], mode = 'curobo';
function vis(f){ return f.traces.map((t,i)=> i<f.n_scene ? true : (i<f.n_cu ? (mode!=='wrs') : (mode!=='curobo'))); }
function draw(){ const f = FIGS[cur], c = f.center, r = 0.13;
  const tr = f.traces.map((t,i)=>Object.assign({}, t, {visible: vis(f)[i]}));
  Plotly.react('plot', tr, {margin:{l:0,r:0,t:30,b:0}, title: cur.replace(/_/g,' ') + ' - ' + META.part + ' grasp ' + META.grasp,
    scene:{aspectmode:'cube', xaxis:{range:[c[0]-r,c[0]+r]}, yaxis:{range:[c[1]-r,c[1]+r]}, zaxis:{range:[Math.max(c[2]-r,-0.01),c[2]+r]}}});
  document.getElementById('verdict').textContent = 'verdict - cuRobo: ' + (f.verdict.gpu?'free':'COLLISION') + ', WRS: ' + (f.verdict.wrs?'free':'COLLISION'); }
function show(m){ mode = m; draw(); }
document.getElementById('cfgs').innerHTML = Object.keys(FIGS).map(k=>`<button onclick="cur='${k}';draw()">${k.replace(/_/g,' ')}</button>`).join('');
draw();
</script></body></html>"""
    page = page.replace("__DATA__", json.dumps(figs)).replace("__META__", json.dumps(meta, default=str))
    (OUT / "collision_models.html").write_text(page, encoding="utf-8")
    print("->", OUT / "collision_models.html")


if __name__ == "__main__":
    main()
