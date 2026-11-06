"""Experiment 1: GPU sphere collision checking (cuRobo, batched) vs WRS mesh collision checking.

    & $py -X utf8 -m joint_search.exp_collision_bench [--items 300]

Scenes (triangular frame; the arm handles part p of the step):
  A  empty         nothing installed, nothing else on the table              p = post_A, start HOME
  B  crowded table step 1: the 5 other parts on the table, 1 cm apart,       p = post_A, start HOME
                   as close to the assembly site as possible
  C  mid assembly  step 5: posts + beam_AB installed, beam_CA on the table   p = beam_BC
                   next to the site
  D  last insert   step 6: the 5 other parts installed (4-7 mm beam gaps)    p = beam_CA
  E  rack10        10 parts on the table, random joint configurations (speed / scaling only)
Items per frame scene (from the search's own IK table, as the search checks them):
  insertion moment   ins(g)            gripper closed to the grasp's jaw, held part not checked
  pick moment        pick(x, g)        gripper closed, p itself not an obstacle
  transit            start -> above pick(x, g), gripper open, p at x is an obstacle
  transfer           above pick -> above insert, holding p, gripper closed
  (lines interpolated every 0.02 rad on both sides)
GPU: lazy_cost.checker.GpuChecker (robot spheres density 2.0 + 1 mm, part covers 6/4 mm).
WRS: joint_search.wrs_check.MeshWorld (triangle meshes, ODE, same jaw widths, held part moved
with the TCP).
Speed: configurations of the scene in batches of 1 ... 100 000 (WRS measured up to 1 000 states,
larger batches extrapolated from its per-state time and marked as such).
Output: joint_search/_output/bench/ (json, markdown tables, figures).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

OUT = Path(__file__).resolve().parent / "_output" / "bench"
BATCHES = (1, 100, 1000, 10000, 100000)


def place_near(sc, parts, cat, gap, avoid=(), exclude_keys=None):
    """Greedy: each part at its catalogue candidate closest to the assembly site, >= gap apart."""
    polys, table = [sc.polygons[q][k] for q, k in avoid], {}
    for q in parts:
        keys = [c for c, _ in sc.candidates[q] if (cat is None or c in cat["pick"][q])
                and all(sc.polygons[q][c].distance(o) >= gap for o in polys)]
        keys.sort(key=lambda c: np.linalg.norm(sc.pose[q][c][:2, 3] - sc.site_xy))
        if keys:
            table[q] = keys[0]
            polys.append(sc.polygons[q][keys[0]])
    return table


def scene_figure(sc, installed, table, p, xs, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import ConvexHull
    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_subplot(1, 2, 1)
    c, s = sc.table_center, sc.table_size
    ax.add_patch(Rectangle((c[0] - s[0] / 2, c[1] - s[1] / 2), s[0], s[1], fc="#f2efe8", ec="#999"))

    def foot(pid, T, **kw):
        v = np.asarray(sc.meshes[pid].vertices) @ T[:3, :3].T + T[:3, 3]
        h = ConvexHull(v[:, :2])
        ax.add_patch(Polygon(v[h.vertices, :2], **kw))
        return v[:, :2].mean(0)
    foot("base", sc.goal["base"], fc="#ccc", ec="#666", alpha=0.6)
    for q in installed:
        m = foot(q, sc.goal[q], fc="#4c78a8", ec="k", alpha=0.85)
        ax.text(*m, q, ha="center", va="center", fontsize=7, color="w")
    for q, k in table.items():
        m = foot(q, sc.pose[q][k], fc="#f58518", ec="k", alpha=0.85)
        ax.text(*m, q, ha="center", va="center", fontsize=7)
    for x in xs:
        foot(p, sc.pose[p][x], fc="none", ec="#54a24b", ls=":", lw=0.8)
    ax.plot(0, 0, "k^", ms=9)
    ax.set_aspect("equal"); ax.autoscale(); ax.set_title(f"{title}\nblue = installed, orange = on the table, green dotted = candidate poses of {p}", fontsize=9)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    for pid, T, col in ([("base", sc.goal["base"], "#bbbbbb")] + [(q, sc.goal[q], "#4c78a8") for q in installed]
                        + [(q, sc.pose[q][k], "#f58518") for q, k in table.items()]):
        m = sc.meshes[pid]
        v = np.asarray(m.vertices) @ T[:3, :3].T + T[:3, 3]
        ax3.add_collection3d(Poly3DCollection(v[np.asarray(m.faces)], facecolor=col, edgecolor="none", alpha=0.9))
    ax3.set_xlim(-0.05, 0.55); ax3.set_ylim(-0.3, 0.3); ax3.set_zlim(0, 0.3)
    ax3.set_box_aspect((0.6, 0.6, 0.3)); ax3.view_init(28, -60); ax3.set_title("3D view (parts)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", type=int, default=300, help="items per frame scene (split over 4 kinds)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None, help="scene letters to evaluate (others only draw their samples)")
    ap.add_argument("--no-speed", action="store_true")
    args = ap.parse_args(argv)
    only = set(args.only) if args.only else None

    import torch
    from lazy_cost.checker import GpuChecker
    from lazy_cost.scene import load_scene
    from .problem import HOME, Problem
    from .wrs_check import MeshWorld
    OUT.mkdir(parents=True, exist_ok=True)
    sync = torch.cuda.synchronize
    rng = np.random.default_rng(args.seed)

    pb = Problem("triangular_frame6")
    sc, cat, chk = pb.scene, pb.cat, pb.chk
    W = MeshWorld(pb)
    G = sc.grasps()
    ids = list(sc.ids)
    ret = lambda q: cat["goal"][q][sorted(cat["goal"][q])[0]]["retreat"]
    scenes = {
        "A_empty": dict(installed=[], others=[], p="post_A", start=HOME),
        "B_crowded_table": dict(installed=[], others=[q for q in ids if q != "post_A"], p="post_A", start=HOME),
        "C_mid_assembly": dict(installed=["post_A", "post_B", "post_C", "beam_AB"], others=["beam_CA"], p="beam_BC",
                               start=ret("beam_AB")),
        "D_last_insert": dict(installed=[q for q in ids if q != "beam_CA"], others=[], p="beam_CA", start=ret("beam_BC")),
    }
    results, speed = {}, {}
    for name, S in scenes.items():
        p, installed = S["p"], S["installed"]
        table = place_near(sc, S["others"], cat, 0.01)
        xs = [x for x in cat["pick"][p] if all(sc.polygons[p][x].distance(sc.polygons[q][k]) >= 0.01 for q, k in table.items())]
        scene_figure(sc, installed, table, p, xs, name, OUT / f"scene_{name}.png")
        placed = [(f"{q}@{k}", q, sc.pose[q][k]) for q, k in table.items()]
        tbl = {q: sc.pose[q][k] for q, k in table.items()}
        pairs = [(x, g) for x in xs for g in cat["pick"][p][x]]
        gl = sorted({g for _, g in pairs})
        k4 = args.items // 4
        G_ins = [gl[i] for i in rng.choice(len(gl), min(k4, len(gl)), replace=False)]
        P_s = [pairs[i] for i in rng.choice(len(pairs), min(3 * k4, len(pairs)), replace=False)]
        P_pick, P_tr, P_tf = P_s[:k4], P_s[k4:2 * k4], P_s[2 * k4:]
        jaw = pb.jaw[p]
        items = ([dict(kind="insertion moment", g=g) for g in G_ins] + [dict(kind="pick moment", x=x, g=g) for x, g in P_pick]
                 + [dict(kind="transit", x=x, g=g) for x, g in P_tr] + [dict(kind="transfer", x=x, g=g) for x, g in P_tf])
        if only is not None and name[0] not in only:
            continue
        # ---- WRS ----------------------------------------------------------------------------------
        for it in items:
            g = it["g"]
            t0 = perf_counter()
            if it["kind"] == "insertion moment":
                W.set(installed, tbl, jaw=jaw[g]); q = cat["goal"][p][g]["ins"]; ok = W.free([q, q])
            elif it["kind"] == "pick moment":
                W.set(installed, tbl, jaw=jaw[g]); q = cat["pick"][p][it["x"]][g]["pick"]; ok = W.free([q, q])
            elif it["kind"] == "transit":
                W.set(installed, dict(tbl, **{p: sc.pose[p][it["x"]]})); ok = W.free([S["start"], cat["pick"][p][it["x"]][g]["lift"]])
            else:
                e = cat["pick"][p][it["x"]][g]
                W.set(installed, tbl, (p, G[p][g][0], e["lift"]), jaw=jaw[g]); ok = W.free([e["lift"], cat["goal"][p][g]["preins"]])
            it["wrs"], it["wrs_why"], it["wrs_ms"] = bool(ok[0]), ok[1], 1e3 * (perf_counter() - t0)
        W.set([], {})
        # ---- GPU (one batched call per kind) --------------------------------------------------------
        chk.set_world(chk.world_dict(installed, placed))
        part = pb.held[p]
        sync(); t0 = perf_counter()
        sel = lambda k: [it for it in items if it["kind"] == k]
        I, Pm, Tr, Tf = sel("insertion moment"), sel("pick moment"), sel("transit"), sel("transfer")
        xs_tr = sorted({it["x"] for it in Tr})
        gs_tf = sorted({it["g"] for it in Tf})
        out = chk.batch_free(
            [(np.stack([cat["goal"][p][it["g"]]["ins"] for it in I]), None, -1, -1, np.array([jaw[it["g"]] for it in I])),
             (np.stack([cat["pick"][p][it["x"]][it["g"]]["pick"] for it in Pm]), None, -1, -1, np.array([jaw[it["g"]] for it in Pm])),
             (np.broadcast_to(S["start"], (len(Tr), 6)), np.stack([cat["pick"][p][it["x"]][it["g"]]["lift"] for it in Tr]), -1,
              np.array([xs_tr.index(it["x"]) for it in Tr])),
             (np.stack([cat["pick"][p][it["x"]][it["g"]]["lift"] for it in Tf]), np.stack([cat["goal"][p][it["g"]]["preins"] for it in Tf]),
              np.array([gs_tf.index(it["g"]) for it in Tf]), -1, np.array([jaw[it["g"]] for it in Tf]))],
            held_bank=[(part, G[p][g][0]) for g in gs_tf], extra_bank=[chk.obstacle_spheres(part, sc.pose[p][x]) for x in xs_tr], step=0.02)
        sync(); gpu_ms = 1e3 * (perf_counter() - t0)
        for grp, flags in zip((I, Pm, Tr, Tf), out):
            for it, f in zip(grp, flags):
                it["gpu"] = bool(f)
        # ---- confusion per kind ---------------------------------------------------------------------
        conf = {}
        for k in ("insertion moment", "pick moment", "transit", "transfer", "all"):
            L = items if k == "all" else sel(k)
            conf[k] = dict(n=len(L), both_free=sum(it["gpu"] and it["wrs"] for it in L), both_hit=sum(not it["gpu"] and not it["wrs"] for it in L),
                           missed=sum(it["gpu"] and not it["wrs"] for it in L), false_hit=sum(not it["gpu"] and it["wrs"] for it in L))
        results[name] = dict(p=p, installed=installed, table=table, candidates=len(xs), obstacles=len(installed) + len(table),
                             confusion=conf, wrs_ms_total=sum(it["wrs_ms"] for it in items), gpu_ms_total=gpu_ms)
        c = conf["all"]
        print(f"{name}: {len(items)} items | agree {(c['both_free'] + c['both_hit']) / c['n']:.0%}, missed {c['missed']}, "
              f"false hit {c['false_hit']} | WRS {results[name]['wrs_ms_total'] / 1e3:.1f} s, GPU {gpu_ms:.0f} ms", flush=True)
        # ---- speed vs batch size: configurations of this scene (moments + line states) --------------
        Q = np.concatenate([np.stack([cat["goal"][p][g]["ins"] for g in gl]),
                            np.stack([cat["pick"][p][x][g]["pick"] for x, g in pairs])])
        J = np.concatenate([[jaw[g] for g in gl], [jaw[g] for _, g in pairs]])
        results[name]["items"] = items
        if not args.no_speed:
            speed[name] = run_speed(chk, W, Q, J, installed, tbl, sync, obstacles=len(installed) + len(table))
    # ---- E: rack10, random configurations, speed + agreement ----------------------------------------
    del pb, chk
    torch.cuda.empty_cache()
    sr = load_scene("rack10")
    chk_r = GpuChecker(sr)
    Wr = MeshWorld(SimpleNamespace(scene=sr))
    table = place_near(sr, list(sr.ids), None, 0.01)
    scene_figure(sr, [], table, sr.ids[0], [], "E_rack10 (10 parts on the table)", OUT / "scene_E_rack10.png")
    placed = [(f"{q}@{k}", q, sr.pose[q][k]) for q, k in table.items()]
    tbl = {q: sr.pose[q][k] for q, k in table.items()}
    lo, hi = (x.cpu().numpy() for x in chk_r.chk.kinematics.get_joint_limits().position)
    Q = rng.uniform(lo, hi, (2000, 6))
    chk_r.set_world(chk_r.world_dict([], placed))
    g_ok = chk_r.batch_free([(Q[:args.items], None, -1, -1)])[0]
    Wr.set([], tbl)
    t0 = perf_counter()
    w_res = [Wr.free([q, q]) for q in Q[:args.items]]
    w_ok = np.array([r[0] for r in w_res])
    wr_ms = 1e3 * (perf_counter() - t0)
    results["E_rack10"] = dict(obstacles=len(table), table=table, confusion={"random configurations": dict(
        n=len(w_ok), both_free=int(np.sum(g_ok & w_ok)), both_hit=int(np.sum(~g_ok & ~w_ok)), missed=int(np.sum(g_ok & ~w_ok)),
        false_hit=int(np.sum(~g_ok & w_ok)))}, wrs_ms_total=wr_ms,
        missed=[dict(q=Q[i].tolist(), wrs_why=w_res[i][1]) for i in np.flatnonzero(g_ok & ~w_ok)])
    print(f"E_rack10: {results['E_rack10']['confusion']}", flush=True)
    if args.no_speed:
        (OUT / "collision_bench_items.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
        return
    speed["E_rack10"] = run_speed(chk_r, Wr, Q, None, [], tbl, sync, obstacles=len(table))
    (OUT / "collision_bench.json").write_text(json.dumps(dict(results=results, speed=speed), indent=1, default=float), encoding="utf-8")
    figures(speed)
    tables(results, speed)


def run_speed(chk, W, Q, J, installed, tbl, sync, obstacles):
    """Per batch size: GPU time (measured) and WRS time (measured up to 1000, extrapolated beyond)."""
    out = dict(obstacles=obstacles, gpu_ms={}, wrs_ms={}, wrs_extrapolated={})
    W.set(installed, tbl)
    t0 = perf_counter()
    nw = min(1000, len(Q))
    for q in Q[:nw]:
        W.free([q, q])
    wrs_per = (perf_counter() - t0) / nw
    W.set([], {})
    for b in BATCHES:
        idx = np.arange(b) % len(Q)
        grp = (Q[idx], None, -1, -1) + ((J[idx],) if J is not None else ())
        chk.batch_free([grp]); sync()                                    # warm-up at this size
        reps = 5 if b <= 10000 else 2
        t0 = perf_counter()
        for _ in range(reps):
            chk.batch_free([grp])
        sync()
        out["gpu_ms"][b] = 1e3 * (perf_counter() - t0) / reps
        out["wrs_ms"][b] = 1e3 * wrs_per * b
        out["wrs_extrapolated"][b] = b > nw
    out["wrs_ms_per_state"] = 1e3 * wrs_per
    out["gpu_us_per_state_at_100k"] = 1e3 * out["gpu_ms"][100000] / 100000
    print(f"  speed ({obstacles} obstacles): WRS {1e3 * wrs_per:.2f} ms/state | GPU " +
          ", ".join(f"{b}: {out['gpu_ms'][b]:.1f} ms" for b in BATCHES), flush=True)
    return out


def figures(speed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, s in speed.items():
        b = np.array(BATCHES)
        l, = ax[0].plot(b, [s["gpu_ms"][k] for k in BATCHES], "-o", label=f"GPU {name}")
        ax[0].plot(b, [s["wrs_ms"][k] for k in BATCHES], "--s", color=l.get_color(), alpha=0.6, label=f"WRS {name}")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("configurations per call")
    ax[0].set_ylabel("time (ms)"); ax[0].set_title("Collision check time vs batch size (WRS dashed; >1000 extrapolated)", fontsize=9)
    ax[0].legend(fontsize=6, ncol=2); ax[0].grid(True, which="both", alpha=0.3)
    names = sorted(speed, key=lambda n: speed[n]["obstacles"])
    obs = [speed[n]["obstacles"] for n in names]
    ax[1].plot(obs, [speed[n]["wrs_ms_per_state"] * 1e3 for n in names], "--s", label="WRS mesh (CPU)")
    ax[1].plot(obs, [speed[n]["gpu_us_per_state_at_100k"] for n in names], "-o", label="GPU spheres (batch 100 000)")
    for n, o in zip(names, obs):
        ax[1].annotate(n.split("_")[0], (o, speed[n]["wrs_ms_per_state"] * 1e3), fontsize=7)
    ax[1].set_yscale("log"); ax[1].set_xlabel("parts in the scene (installed + on the table)"); ax[1].set_ylabel("time per configuration (us)")
    ax[1].set_title("Time per configuration vs number of obstacles", fontsize=9); ax[1].legend(fontsize=8); ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "speed.png", dpi=130); plt.close(fig)


def tables(results, speed):
    L = ["# Experiment 1: GPU sphere vs WRS mesh collision checking", "",
         "## Agreement (reference: WRS mesh)", "",
         "| scene | item | n | both free | both collide | missed (GPU free, WRS hit) | false hit (GPU hit, WRS free) | agreement |",
         "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        for k, c in r["confusion"].items():
            L.append(f"| {name} | {k} | {c['n']} | {c['both_free']} | {c['both_hit']} | {c['missed']} | {c['false_hit']} | "
                     f"{(c['both_free'] + c['both_hit']) / max(c['n'], 1):.0%} |")
    L += ["", "## Speed", "", "| scene | obstacles | WRS per configuration | " + " | ".join(f"GPU {b}" for b in BATCHES) +
          " | GPU per configuration at 100 000 | speed-up at 100 000 |", "|---|---|---|" + "---|" * (len(BATCHES) + 2)]
    for name, s in speed.items():
        L.append(f"| {name} | {s['obstacles']} | {s['wrs_ms_per_state']:.3f} ms | " + " | ".join(f"{s['gpu_ms'][b]:.1f} ms" for b in BATCHES)
                 + f" | {s['gpu_us_per_state_at_100k']:.2f} us | x{s['wrs_ms'][100000] / s['gpu_ms'][100000]:.0f} |")
    (OUT / "collision_bench.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
