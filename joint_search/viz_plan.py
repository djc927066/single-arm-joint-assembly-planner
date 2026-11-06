"""Replay of a run_frame result as an offline HTML page (three.js): the robot executes the chosen plan,
segment by segment, in the forward order of joint_search.validate.forward_steps.

    & $py -X utf8 -m joint_search.viz_plan joint_search/_output/frame_<tag>_<time>.json [--fps 30]
-> the same path with "_viewer.html".

Free segments whose straight joint line is collision-free (our checker) are drawn along that line,
the others along a fresh cuRobo path (as the validation); every piece is timed with TOPP
(lazy_cost.timing), so the replay time is close to the validated time. The problem must be built with
the settings stored in the run (same defaults), so the IK catalogue is read from the cache.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

GRIP_S = 0.5            # s shown for closing / opening the gripper (not part of the plan time)
JAW_OPEN = 0.085        # fully open jaw of the render meshes (= lazy_cost.checker.GpuChecker.JAW_OPEN)
CONTACT_OFFSET = 0.01   # WRS grasp planning contact_offset (lazy_cost.gen_grasps): jaw = contact width + 2x
COLORS = ["#e15759", "#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#edc948", "#76b7b2", "#ff9da7"]


def _b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def _mesh(v, f):
    return dict(v=_b64(np.asarray(v), "<f4"), f=_b64(np.asarray(f), "<u4"))


def build(run_path: Path, fps=30, fingers_view="plan"):
    from curobo_bridge.runtime import ASSET_DIR, prepare
    prepare()
    import trimesh
    from scipy.spatial.transform import Rotation
    from curobo_bridge.export_model import fk_frames, load_chain
    from lazy_cost.refine import CuroboRefiner
    from lazy_cost.timing import topp
    from .problem import HOME, Problem
    from .search import Node
    from .validate import Validator, forward_steps

    run = json.loads(run_path.read_text(encoding="utf-8"))
    a = run["args"]
    pb = Problem(a["model"], penalty=a.get("penalty"), grasps_per_pose=a.get("grasps_per_pose", 8),
                 sampler=a.get("sampler", "fps"), placements=a.get("placements", "sealp"), orient=a.get("orient", "first"),
                 grid=a.get("grid", "fine"), grid_step=a.get("grid_step", 0.015), keepout=a.get("keepout", "sealp"),
                 home_clear=a.get("home_clear", 0.03),
                 center=None if a.get("goal_pos") == "model" else tuple(float(v) for v in a.get("goal_pos", "0.373,0,0").split(",")))
    sc, cat, chk = pb.scene, pb.cat, pb.chk
    steps = [tuple(s) for s in run["chosen"]["steps"]]
    node = Node(R=frozenset(), suffix=tuple(steps))
    ref = CuroboRefiner(chk, detax=bool(a.get("detax")))
    for p in pb.ids:
        ref.fit(p)
    val = Validator(pb, ref)

    def free_path(s):
        """[a, b] when the straight line is free, else a cuRobo path (as the validation)."""
        if val._straight_free(s):
            return np.stack([s["a"], s["b"]]), "straight"
        world = chk.world_dict(s["installed"], pb.placed(s["table"]))
        if s["kind"] == "transfer":
            p, g = s["part"], s["g"]
            ref.set_world(world)
            ref._attach(p, sc.grasps()[p][g][0], s["a"], val._jaw(s))
            P = ref._plan(ref.mp["transfer"], [(s["a"], s["b"])])[0]
            ref.mp["transfer"].attachment_manager.detach(link_name="attached_object")
        else:
            if s["kind"] == "transit":
                world = dict(world, mesh=dict(world["mesh"], **{s["part"]: chk.mesh_entry(s["part"], sc.pose[s["part"]][s["x"]])}))
            ref.set_world(world)
            P = ref._plan(ref.mp["transit"], [(s["a"], s["b"])])[0]
        if P is None:
            return np.stack([s["a"], s["b"]]), "failed"
        return np.asarray(P, float), "planned"

    # ---- timeline: (label, part, path, mode, held part, installed after) --------------------------
    LABEL = dict(transit="空手移动", transfer="搬运", ret="回 HOME", down="竖直下降", up="竖直抬起",
                 insert="竖直插入", out="松手后上退", close="夹紧", open="松开")
    pieces = []
    segs = forward_steps(pb, node)
    by = {}
    for s in segs:
        by.setdefault(s["part"], []).append(s)
    for k, (p, x, g) in enumerate(steps):
        e, v = cat["pick"][p][x][g], cat["goal"][p][g]
        tr = next(s for s in by[p] if s["kind"] == "transit")
        tf = next(s for s in by[p] if s["kind"] == "transfer")
        P, how = free_path(tr)
        pieces.append(("transit", p, P, how, None))
        pieces.append(("down", p, np.stack([e["hover"], e["pick"]]), "fixed", None))
        pieces.append(("close", p, np.stack([e["pick"], e["pick"]]), "grip", None))
        pieces.append(("up", p, np.stack([e["pick"], e["lift"]]), "fixed", p))
        P, how = free_path(tf)
        pieces.append(("transfer", p, P, how, p))
        pieces.append(("insert", p, np.stack([v["preins"], v["ins"]]), "fixed", p))
        pieces.append(("open", p, np.stack([v["ins"], v["ins"]]), "grip", p))
        pieces.append(("out", p, np.stack([v["ins"], v["retreat"]]), "fixed", None))
    last = next(s for s in segs if s["kind"] == "return")
    P, how = free_path(last)
    pieces.append(("ret", None, P, how, None))

    # ---- frames ---------------------------------------------------------------------------------
    chain = load_chain(ASSET_DIR / "panthera_full")
    links = chain["collision_links"]
    parent = {"base_link": -1, "link1": 0, "link2": 1, "link3": 2, "link4": 3, "link5": 4,
              "gripper_palm": 5, "finger_left": 5, "finger_right": 5}
    base = np.asarray(chain["base"], float)
    tcp_rel = np.asarray(chain["tcp"], float)
    G = {p: sc.grasps()[p][g][0] for p, _, g in steps}
    init = {p: sc.pose[p][x] for p, x, _ in steps}
    # consistency: TCP of the render chain vs the planner's checker
    import torch
    qs = np.stack([cat["pick"][p][x][g]["pick"] for p, x, g in steps])
    _, Rm, pt = chk.kinematics(torch.as_tensor(qs, dtype=torch.float32, device="cuda"))
    err = max(float(np.abs((fk_frames(chain, q)[5] @ tcp_rel)[:3, 3] - pt[i].cpu().numpy()).max()) for i, q in enumerate(qs))
    print(f"TCP render chain vs checker: max position error {1e3 * err:.2f} mm", flush=True)

    # fingers: the render meshes are fully open; per frame the closing fraction (0 open, 1 closed to
    # the grasp's jaw width, as the planner's checker) and the TCP y axis (closing axis) move them
    jaw = {p: float(cat["jaw"][p][g]) for p, _, g in steps}
    fingers = []
    frames, events, t = [], [], 0.0
    done = set()
    order = [p for p, _, _ in steps]
    for kind, p, P, how, held in pieces:
        if how == "grip":
            pts, ts = P, np.array([0.0, GRIP_S])
        else:
            pts, ts = topp(P)
        dur = float(ts[-1])
        n = max(2, int(round(dur * fps)) + 1)
        tt = np.linspace(0, dur, n)
        Q = np.stack([np.interp(tt, ts, pts[:, j]) for j in range(6)], 1)
        events.append(dict(t0=t, t1=t + dur, kind=kind, label=LABEL[kind], part=p or "-",
                           how=how, step=order.index(p) + 1 if p else len(order) + 1))
        for i, q in enumerate(Q[:-1]):
            fr = fk_frames(chain, q)
            poses = [base if parent[nm] < 0 else fr[parent[nm]] for nm in links]
            tcp = fr[5] @ tcp_rel
            s = i / max(len(Q) - 2, 1)
            cf = s if kind == "close" else (1 - s if kind == "open" else float(held is not None))
            fingers.append([cf, jaw[p] if p else JAW_OPEN, *tcp[:3, 1]])
            for q2 in order:
                if q2 in done:
                    poses.append(sc.goal[q2])
                elif q2 == held:
                    poses.append(tcp @ np.linalg.inv(G[q2]))
                else:
                    poses.append(init[q2])
            frames.append(poses)
        if kind == "open":
            done.add(p)
        t += dur
    fr = fk_frames(chain, pieces[-1][2][-1])
    frames.append([base if parent[nm] < 0 else fr[parent[nm]] for nm in links] + [sc.goal[q2] for q2 in order])
    fingers.append([0.0, JAW_OPEN, *(fr[5] @ tcp_rel)[:3, 1]])
    arr = np.zeros((len(frames), len(frames[0]), 7), np.float32)
    for i, poses in enumerate(frames):
        for j, h in enumerate(poses):
            arr[i, j, :3] = h[:3, 3]
            arr[i, j, 3:] = Rotation.from_matrix(h[:3, :3]).as_quat()

    bodies = []
    for nm in links:
        m = trimesh.load(ASSET_DIR / "panthera_full" / "meshes" / f"{nm}.stl")
        v, f = np.asarray(m.vertices), np.asarray(m.faces)
        if len(f) > 6000:                                   # decimate like curobo_bridge.assembly_demo.export_viz
            try:
                import open3d as o3d
                om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
                om = om.simplify_quadric_decimation(target_number_of_triangles=6000)
                v, f = np.asarray(om.vertices), np.asarray(om.triangles)
            except ImportError:
                pass
        bodies.append(dict(name=nm, kind="robot", **_mesh(v, f)))
    for i, p in enumerate(order):
        bodies.append(dict(name=p, kind="part", color=COLORS[i % len(COLORS)], **_mesh(sc.meshes[p].vertices, sc.meshes[p].faces)))
    statics = [dict(name="base", color="#8fa9c7", pose=np.asarray(sc.goal["base"]).T.reshape(-1).tolist(),
                    **_mesh(sc.meshes["base"].vertices, sc.meshes["base"].faces))]
    data = dict(
        bodies=bodies, statics=statics, fps=fps, frames=len(frames), duration=t, poses=_b64(arr, "<f4"),
        fingers=_b64(np.asarray(fingers), "<f4"), finger_idx=[links.index("finger_left"), links.index("finger_right")],
        jaw_open=JAW_OPEN, contact_offset=CONTACT_OFFSET,
        events=events, table=dict(center=sc.table_center.tolist(), size=sc.table_size.tolist()),
        plan=[dict(step=i + 1, part=p, pose=x, grasp=g, color=COLORS[i % len(COLORS)]) for i, (p, x, g) in enumerate(steps)],
        result=dict(time=run["chosen"].get("time", run["chosen"].get("confirmed")), est=run["chosen"]["est"], lb=run["chosen"]["lb"],
                    validated="time" in run["chosen"],
                    search=a.get("search", "eager"), source=run_path.name))
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    if fingers_view == "contact":           # fingers shown on the part by default (the menu still switches)
        html = html.replace('<option value="plan" selected>', '<option value="plan">').replace(
            '<option value="contact">', '<option value="contact" selected>')
    out = run_path.with_name(run_path.stem + "_viewer.html")
    out.write_text(html, encoding="utf-8")
    print(f"replay {t:.2f} s (plan pieces timed with TOPP, gripper {GRIP_S} s each shown extra), "
          f"{len(frames)} frames -> {out}", flush=True)
    return out


TEMPLATE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>单臂装配回放</title>
<style>
:root{--bg:#f6f5f1;--panel:#fff;--ink:#1f2328;--mute:#6b7280;--line:#e5e2da;--acc:#2f6fb3}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#15171a;--panel:#1e2125;--ink:#e8e6e1;--mute:#9aa0a6;--line:#30343a;--acc:#6ea8e8}}
:root[data-theme=dark]{--bg:#15171a;--panel:#1e2125;--ink:#e8e6e1;--mute:#9aa0a6;--line:#30343a;--acc:#6ea8e8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,"Microsoft YaHei",sans-serif}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:12px;padding:12px;height:100vh}
@media (max-width:900px){.wrap{grid-template-columns:1fr;height:auto}#view{height:60vh}}
#view{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;min-height:360px}
#view canvas{display:block;width:100%;height:100%}
.hud{position:absolute;left:12px;top:10px;font-size:15px;font-weight:600;text-shadow:0 0 4px var(--panel)}
.hud small{display:block;font-weight:400;color:var(--mute)}
.side{display:flex;flex-direction:column;gap:10px;overflow:auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.card h2{font-size:13px;margin:0 0 6px;color:var(--mute);font-weight:600}
.ctl{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button,select{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:6px;padding:4px 10px;cursor:pointer}
input[type=range]{width:100%}
table{border-collapse:collapse;width:100%;font-size:12.5px}td,th{padding:3px 4px;border-bottom:1px solid var(--line);text-align:left}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
tr.on{background:color-mix(in srgb,var(--acc) 15%,transparent)}
.big{font-size:20px;font-weight:700}
.mute{color:var(--mute)}
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<div class="wrap">
 <div id="view"><div class="hud"><span id="now">-</span><small id="sub"></small></div></div>
 <div class="side">
  <div class="card"><h2>结果</h2><div class="big" id="tt"></div><div class="mute" id="meta"></div></div>
  <div class="card"><h2>回放</h2>
   <div class="ctl"><button id="play">暂停</button>
    <select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select>
    <span id="clock" class="mute"></span></div>
   <div class="ctl" style="margin-top:6px"><span class="mute">手指</span>
    <select id="jaw"><option value="plan" selected>规划用的宽度（grasp 记录值）</option><option value="contact">贴住零件（记录值 − 2 cm）</option></select></div>
   <input id="seek" type="range" min="0" max="1000" value="0"></div>
  <div class="card"><h2>装配顺序与初始位置</h2><table id="plan"><thead><tr><th>#</th><th>零件</th><th>初始位置 (x, y)</th><th>姿态</th></tr></thead><tbody></tbody></table></div>
  <div class="card"><h2>当前段</h2><table id="ev"><tbody></tbody></table></div>
 </div>
</div>
<script type="module">
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
const D = __DATA__;
const b64 = (s, T) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const view = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({antialias: true}); renderer.setPixelRatio(devicePixelRatio);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(40, 1, 0.01, 20); cam.up.set(0, 0, 1); cam.position.set(0.95, -0.95, 0.85);
const ctr = new OrbitControls(cam, renderer.domElement); ctr.target.set(0.25, 0, 0.08); ctr.update();
scene.add(new THREE.HemisphereLight(0xffffff, 0x888888, 1.6));
const dl = new THREE.DirectionalLight(0xffffff, 1.6); dl.position.set(1, -1, 2); scene.add(dl);
function geo(b) { const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(b64(b.v, Float32Array), 3)); g.setIndex(new THREE.BufferAttribute(b64(b.f, Uint32Array), 1)); g.computeVertexNormals(); return g; }
const T = D.table; const tg = new THREE.BoxGeometry(...T.size);
const table = new THREE.Mesh(tg, new THREE.MeshStandardMaterial({color: 0xd8cfbf, roughness: 0.9})); table.position.set(...T.center); scene.add(table);
const grid = new THREE.GridHelper(T.size[0], 12, 0xb9ae9a, 0xcfc6b6); grid.rotation.x = Math.PI / 2; grid.position.set(T.center[0], T.center[1], 0.0005); scene.add(grid);
for (const s of D.statics) { const m = new THREE.Mesh(geo(s), new THREE.MeshStandardMaterial({color: s.color, roughness: 0.7})); m.matrixAutoUpdate = false; m.matrix.fromArray(s.pose); scene.add(m); }
const meshes = D.bodies.map(b => { const m = new THREE.Mesh(geo(b), new THREE.MeshStandardMaterial({color: b.kind === "robot" ? 0x9aa3ad : b.color, roughness: 0.55, metalness: b.kind === "robot" ? 0.25 : 0})); scene.add(m); return m; });
const P = b64(D.poses, Float32Array), nb = D.bodies.length;
const F = b64(D.fingers, Float32Array), jawMode = document.getElementById("jaw");
function setFrame(i) {
  for (let j = 0; j < nb; j++) { const o = (i * nb + j) * 7; meshes[j].position.set(P[o], P[o + 1], P[o + 2]); meshes[j].quaternion.set(P[o + 3], P[o + 4], P[o + 5], P[o + 6]); }
  // close the fingers along the TCP y axis: to the grasp's jaw width (planner) or onto the part
  const o = i * 5, w = F[o + 1] - (jawMode.value === "contact" ? 2 * D.contact_offset : 0);
  const d = F[o] * Math.max(D.jaw_open - w, 0) / 2;
  D.finger_idx.forEach((j, k) => { const s = k === 0 ? 1 : -1; meshes[j].position.x -= s * d * F[o + 2]; meshes[j].position.y -= s * d * F[o + 3]; meshes[j].position.z -= s * d * F[o + 4]; });
}
const R = D.result;
document.getElementById("tt").textContent = `执行时间 ${R.time.toFixed(3)} s`;
document.getElementById("meta").textContent = `${R.validated ? "全流程验证通过（cuRobo + 网格碰撞检测）" : "旧版结果（cuRobo 确认，未做全流程网格检查）"} · 估计 ${R.est.toFixed(3)} s · 下界 ${R.lb.toFixed(3)} s · 搜索 ${R.search} · ${R.source}`;
const tb = document.querySelector("#plan tbody");
for (const r of D.plan) { const [x, y, , tag] = r.pose.split(","); tb.insertAdjacentHTML("beforeend", `<tr id="row${r.step}"><td>${r.step}</td><td><span class="sw" style="background:${r.color}"></span>${r.part}</td><td>${(+x).toFixed(3)}, ${(+y).toFixed(3)}</td><td>${tag || "goal"}</td></tr>`); }
const ev = document.querySelector("#ev tbody");
let t = 0, playing = true, last = performance.now();
const seek = document.getElementById("seek"), speed = document.getElementById("speed");
document.getElementById("play").onclick = e => { playing = !playing; e.target.textContent = playing ? "暂停" : "播放"; };
seek.oninput = () => { t = seek.value / 1000 * D.duration; };
function show() {
  const i = Math.min(D.frames - 1, Math.round(t * D.fps)); setFrame(i);
  const e = D.events.find(e => t >= e.t0 && t < e.t1) || D.events[D.events.length - 1];
  document.getElementById("now").textContent = e.part === "-" ? `${e.label}` : `第 ${e.step} 步 · ${e.part} · ${e.label}`;
  document.getElementById("sub").textContent = e.how === "planned" ? "这一段直线被挡，按 cuRobo 规划的路径" : (e.how === "failed" ? "cuRobo 没有解出，按直线显示" : "");
  document.getElementById("clock").textContent = `${t.toFixed(2)} / ${D.duration.toFixed(2)} s`;
  seek.value = Math.round(t / D.duration * 1000);
  document.querySelectorAll("#plan tr").forEach(r => r.classList.toggle("on", r.id === `row${e.step}`));
  ev.innerHTML = `<tr><td>段</td><td>${e.label}${e.how === "planned" ? "（规划路径）" : ""}</td></tr><tr><td>时间</td><td>${e.t0.toFixed(2)} – ${e.t1.toFixed(2)} s</td></tr>`;
}
function resize() { const w = view.clientWidth, h = view.clientHeight; renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(view); resize();
function loop(now) { const dt = (now - last) / 1000; last = now; if (playing) { t += dt * +speed.value; if (t > D.duration) t = 0; } show(); renderer.setClearColor(css("--panel") || "#ffffff"); renderer.render(scene, cam); requestAnimationFrame(loop); }
requestAnimationFrame(loop);
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--jaw", choices=("plan", "contact"), default="plan",
                    help="fingers shown at the planner's jaw width (part + 2 cm) or on the part, at start")
    a = ap.parse_args()
    build(Path(a.run), a.fps, a.jaw)
