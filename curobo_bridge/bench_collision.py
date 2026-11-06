"""cuRobo 球模型 vs WRS 碰撞检测：误报 / 漏报 / 耗时。

运行（cuRobo venv，仓库根目录）::

    & E:\\Projects\\wrs-sealp-assemble\\dual_sealp\\cuRobotest\\.venv\\Scripts\\python.exe `
        -X utf8 -m curobo_bridge.bench_collision --scenes 6

两个参照标签（都只看机器人-场景碰撞，WRS 不做自碰撞）：

* ``wrs_prim``：``arm.is_collided(obstacles)``——WRS/RRT 当前实际使用的语义：
  机器人各部件局部 AABB+2 mm 盒 vs 障碍局部 AABB+2 mm 盒（sealp 的 "triangles"
  图元参数无效、回退为 AABB，这里如实复现）。
* ``wrs_mesh``：各外部碰撞部件 ``cmodel.is_mcdwith(obstacles)``——ODE 三角网格求交，
  即几何真值（只判表面相交；完全包含的情况检测不到）。

状态采样：每个场景先均匀采样关节角池，用"机器人表面采样点到障碍的无符号距离"
（open3d，与被测模型无关）分层：一部分均匀抽取（反映真实基率），一部分取距离
< ``band`` 的边界态（最能区分模型的难例）。

cuRobo 被测模型：MorphIt 各密度 + 复刻的 AABB 外接球覆盖（原主线方案），
每个再叠加若干球半径膨胀 ``collision_sphere_buffer``。

漏报（FN）= 参照判碰、cuRobo 判安全，是危险方向；误报（FP）= 参照安全、cuRobo 判碰，
只会丢掉可行方案。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .runtime import ASSET_DIR, OUTPUT_DIR, prepare

JOINT_LIMITS = np.array([[-2.4, 2.4], [0.0, 3.2], [0.0, 4.0],
                         [-1.6, 1.6], [-1.7, 1.7], [-2.5, 2.5]])


def _robot_surface_points(model_dir: Path, per_link: int, rng) -> dict:
    import trimesh
    from .export_model import load_chain
    chain = load_chain(model_dir)
    parent = {"base_link": -1, "link1": 0, "link2": 1, "link3": 2, "link4": 3, "link5": 4,
              "gripper_palm": 5, "finger_left": 5, "finger_right": 5}
    pts = {}
    for n in chain["collision_links"]:
        m = trimesh.load(model_dir / f"meshes/{n}.stl")
        pts[n] = (parent[n], np.asarray(m.sample(per_link, seed=int(rng.integers(1 << 30)))))
    return chain, pts


def proxy_distance(chain, pts, obstacle_meshes, qs) -> np.ndarray:
    """Min unsigned distance of sampled robot surface points to obstacles [m]."""
    import open3d as o3d
    from .export_model import fk_frames

    rs = o3d.t.geometry.RaycastingScene()
    for m in obstacle_meshes:
        rs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices),
                                      o3d.utility.Vector3iVector(m.faces))))
    base = np.asarray(chain["base"], float)
    allp = []
    for q in qs:
        fr = fk_frames(chain, q)
        w = [p @ (base if par < 0 else fr[par])[:3, :3].T + (base if par < 0 else fr[par])[:3, 3]
             for par, p in pts.values()]
        allp.append(np.concatenate(w))
    allp = np.stack(allp).astype(np.float32)
    d = rs.compute_distance(o3d.core.Tensor(allp.reshape(-1, 3))).numpy()
    return d.reshape(len(qs), -1).min(axis=1)


def wrs_labels(arm, wobs, qs, ext_links):
    prim, mesh = np.zeros(len(qs), bool), np.zeros(len(qs), bool)
    t0 = time.perf_counter()
    for i, q in enumerate(qs):
        arm.goto_given_conf(q)
        prim[i] = bool(arm.is_collided(obstacle_list=wobs))
    t1 = time.perf_counter()
    for i, q in enumerate(qs):
        arm.goto_given_conf(q)
        mesh[i] = any(l.cmodel.is_mcdwith(wobs) for l in ext_links)
    t2 = time.perf_counter()
    arm.goto_given_conf(np.zeros(6))
    return prim, mesh, (t1 - t0) / len(qs), (t2 - t1) / len(qs)


def confusion(pred, ref) -> dict:
    pred, ref = np.asarray(pred, bool), np.asarray(ref, bool)
    fn = int(np.sum(ref & ~pred))
    fp = int(np.sum(~ref & pred))
    pos, neg = int(ref.sum()), int((~ref).sum())
    return dict(n=int(len(ref)), positives=pos, fn=fn, fp=fp,
                fn_rate=fn / pos if pos else 0.0, fp_rate=fp / neg if neg else 0.0,
                accuracy=float(np.mean(pred == ref)))


def time_curobo(checker, qs_np, sizes, repeats=20):
    import torch
    from .curobo_models import scene_hits
    out = {}
    for b in sizes:
        q = np.resize(qs_np, (b, 6))
        qt = torch.as_tensor(q, dtype=torch.float32, device="cuda").reshape(b, 1, 6)
        for _ in range(3):
            scene_hits(checker, qt)
        torch.cuda.synchronize()
        res, e2e = [], []
        for _ in range(repeats):
            t = time.perf_counter()
            s, _ = scene_hits(checker, qt)
            torch.cuda.synchronize()
            res.append(time.perf_counter() - t)
            t = time.perf_counter()
            qt2 = torch.as_tensor(q, dtype=torch.float32, device="cuda").reshape(b, 1, 6)
            s, _ = scene_hits(checker, qt2)
            s.cpu().numpy()
            e2e.append(time.perf_counter() - t)
        out[str(b)] = dict(resident_ms=1e3 * float(np.median(res)),
                           end_to_end_ms=1e3 * float(np.median(e2e)),
                           per_state_us=1e6 * float(np.median(e2e)) / b)
    return out


def main(argv=None) -> None:
    prepare()
    import torch
    from .curobo_models import (make_collision_checker, model_variants, scene_hits,
                                sphere_count, with_buffer)
    from .export_model import make_wrs_arm, wrs_collision_parts
    from .scenes import random_scene

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="panthera_wrsext")
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument("--pool", type=int, default=6000)
    parser.add_argument("--uniform", type=int, default=300)
    parser.add_argument("--boundary", type=int, default=500)
    parser.add_argument("--band", type=float, default=0.03)
    parser.add_argument("--buffers", type=float, nargs="+", default=[0.0, 0.002, 0.005, 0.01])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    model_dir = ASSET_DIR / args.model
    arm = make_wrs_arm()
    ext_links = [l for _, _, l in wrs_collision_parts(arm, False)]
    variants = model_variants(arm, args.model)
    chain, pts = _robot_surface_points(model_dir, 400, rng)

    all_q, all_prim, all_mesh, all_prox, scene_ids = [], [], [], [], []
    preds = {}
    t_prim, t_mesh = [], []
    checkers = {}
    setup = {}
    timing = None
    for s in range(args.scenes):
        scene = random_scene(1000 + s)
        wobs = scene.wrs_obstacles()
        pool = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], size=(args.pool, 6))
        prox = proxy_distance(chain, pts, scene.trimeshes(), pool)
        near = np.flatnonzero(prox < args.band)
        pick_u = rng.choice(args.pool, size=args.uniform, replace=False)
        pick_b = rng.choice(near, size=min(args.boundary, len(near)), replace=False)
        idx = np.concatenate([pick_u, pick_b])
        qs = pool[idx]
        prim, mesh, tp, tm = wrs_labels(arm, wobs, qs, ext_links)
        t_prim.append(tp)
        t_mesh.append(tm)
        cscene = scene.curobo_scene()
        qt = torch.as_tensor(qs, dtype=torch.float32, device="cuda").reshape(-1, 1, 6)
        for name, robot in variants.items():
            for buf in args.buffers:
                key = f"{name}+{buf * 1000:g}mm"
                if key not in checkers:
                    t = time.perf_counter()
                    checkers[key] = make_collision_checker(with_buffer(robot, buf), cscene)
                    torch.cuda.synchronize()
                    setup[key] = time.perf_counter() - t
                else:
                    checkers[key].update_world(cscene)
                hit, _ = scene_hits(checkers[key], qt)
                preds.setdefault(key, []).append(hit.cpu().numpy())
        if timing is None:
            timing = {k: time_curobo(checkers[k], qs, [1, 32, 256, 2048])
                      for k in (f"morphit_d1+0mm", "aabb20mm+0mm")}
        all_q.append(qs)
        all_prim.append(prim)
        all_mesh.append(mesh)
        all_prox.append(prox[idx])
        scene_ids.append(np.full(len(qs), s))
        print(f"scene {s}: {scene.counts()} states={len(qs)} (boundary {len(pick_b)}) "
              f"prim+={prim.mean():.3f} mesh+={mesh.mean():.3f}", flush=True)

    prim = np.concatenate(all_prim)
    mesh = np.concatenate(all_mesh)
    uniform_mask = np.concatenate([np.r_[np.ones(args.uniform, bool),
                                         np.zeros(len(p) - args.uniform, bool)] for p in all_prim])
    report = dict(
        created=datetime.now().isoformat(timespec="seconds"),
        args=vars(args),
        states=int(len(prim)),
        wrs_prim_ms_per_state=1e3 * float(np.mean(t_prim)),
        wrs_mesh_ms_per_state=1e3 * float(np.mean(t_mesh)),
        wrs_prim_vs_mesh=dict(all=confusion(prim, mesh),
                              uniform=confusion(prim[uniform_mask], mesh[uniform_mask])),
        curobo_setup_s=setup,
        curobo_timing=timing,
        sphere_counts={k: sphere_count(v) for k, v in variants.items()},
        models={},
    )
    for key, chunks in preds.items():
        p = np.concatenate(chunks)
        report["models"][key] = dict(
            vs_mesh=confusion(p, mesh), vs_prim=confusion(p, prim),
            vs_mesh_uniform=confusion(p[uniform_mask], mesh[uniform_mask]))
    out_dir = OUTPUT_DIR / f"collision_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "labels.npz", q=np.concatenate(all_q), prim=prim, mesh=mesh,
                        prox=np.concatenate(all_prox), scene=np.concatenate(scene_ids),
                        **{k.replace("+", "_").replace(".", "p"): np.concatenate(v)
                           for k, v in preds.items()})
    _print(report)
    print("->", out_dir)


def _print(r):
    print(f"\nstates={r['states']}  WRS prim {r['wrs_prim_ms_per_state']:.3f} ms/state, "
          f"WRS mesh {r['wrs_mesh_ms_per_state']:.3f} ms/state")
    c = r["wrs_prim_vs_mesh"]["all"]
    print(f"WRS prim vs mesh: FN={c['fn']} FP={c['fp']} (pos={c['positives']})")
    print(f"{'model':22s} {'sph':>5s} | vs mesh: {'FN':>4s} {'FNr':>6s} {'FP':>4s} {'FPr':>6s} "
          f"| vs prim: {'FN':>4s} {'FP':>4s}")
    for k, m in r["models"].items():
        a, b = m["vs_mesh"], m["vs_prim"]
        n = r["sphere_counts"][k.split("+")[0]]
        print(f"{k:22s} {n:5d} |          {a['fn']:4d} {a['fn_rate']:6.3f} {a['fp']:4d} "
              f"{a['fp_rate']:6.3f} |          {b['fn']:4d} {b['fp']:4d}")
    for k, t in (r["curobo_timing"] or {}).items():
        print(k, {b: f"{v['end_to_end_ms']:.2f}ms ({v['per_state_us']:.1f}us/state)" for b, v in t.items()})


if __name__ == "__main__":
    main()
