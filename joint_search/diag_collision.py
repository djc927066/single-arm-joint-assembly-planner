"""Where do our GPU sphere checks disagree with the WRS mesh checks, and which parameters fix it?

    & $py -X utf8 -m joint_search.diag_collision [--contexts 12] [--per 80]

1. Segments: random contexts as in exp_step_rank (same seed -> the same first scenes); for the part
   of the step, the feasible (pose, grasp) pairs after the search's screening; per context up to
   ``per`` segments: transit (start -> above pick, empty hand, gripper open, the part at x is an
   obstacle) and transfer (above pick -> above insert, holding the part, gripper closed).
2. Reference: WRS mesh check of the straight joint line (wrs_check.MeshWorld.free, every 0.02 rad).
3. Ours, ablated, to see which component flags a line: robot spheres only / + held part cover /
   + cover of the part being picked. Then parameter variants (no new stage, same code path):
   robot sphere buffer, part cover (spacing, radius), robot sphere model density.
Output: joint_search/_output/diag_collision.json + printed tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

OUT = Path(__file__).resolve().parent / "_output"


def surface_cover(mesh, spacing, radius):
    """Same construction as scene.held_spheres, without its per-part cache."""
    import trimesh
    n = int(np.ceil(mesh.area / spacing ** 2))
    pts, _ = trimesh.sample.sample_surface_even(mesh, n, seed=0)
    pts = np.r_[pts, mesh.vertices]
    return np.c_[pts, np.full(len(pts), radius)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contexts", type=int, default=12)
    ap.add_argument("--per", type=int, default=80)
    ap.add_argument("--gap", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--load", action="store_true", help="reuse joint_search/_output/diag_segments.json")
    args = ap.parse_args(argv)

    import torch
    from lazy_cost.checker import GpuChecker
    from .problem import HOME, Problem
    from .wrs_check import MeshWorld
    from .exp_step_rank import fast_rank

    pb = Problem("triangular_frame6")
    sc, cat = pb.scene, pb.cat
    W = MeshWorld(pb)
    G = sc.grasps()
    rng = np.random.default_rng(args.seed)
    segs = []
    t_wrs = 0.0
    done = 0
    seg_file = OUT / "diag_segments.json"
    if args.load:
        D = json.loads(seg_file.read_text(encoding="utf-8"))
        segs, t_wrs, done = D["segments"], D["wrs_s"], args.contexts
    for attempt in range(300):
        if done == args.contexts:
            break
        order = list(sc.orders[int(rng.integers(len(sc.orders)))])
        k = int(rng.integers(len(order)))
        p, installed, later = order[k], order[:k], order[k + 1:]
        table, polys = {}, []
        for q in rng.permutation(later):
            keys = [c for c, _ in sc.candidates[q] if c in cat["pick"][q] and all(sc.polygons[q][c].distance(o) >= args.gap for o in polys)]
            if not keys:
                break
            c = keys[int(rng.integers(len(keys)))]
            table[str(q)] = c
            polys.append(sc.polygons[q][c])
        if len(table) != len(later):
            continue
        xs = [c for c, _ in sc.candidates[p] if c in cat["pick"][p] and all(sc.polygons[p][c].distance(o) >= args.gap for o in polys)]
        prev = order[k - 1] if k else None
        start = HOME if prev is None else cat["goal"][prev][sorted(cat["goal"][prev])[int(rng.integers(len(cat["goal"][prev])))]]["retreat"]
        est, _, scr = fast_rank(pb, p, start, installed, table, xs)
        if not est:
            continue
        keys = list(est)
        pick = rng.choice(len(keys), min(len(keys), args.per // 2), replace=False)
        tbl = {q: sc.pose[q][c] for q, c in table.items()}
        for i in pick:
            x, g = keys[i]
            e, v = cat["pick"][p][x][g], cat["goal"][p][g]
            t0 = perf_counter()
            W.set(installed, dict(tbl, **{p: sc.pose[p][x]}))
            ok_a = W.free([start, e["lift"]])
            W.set(installed, tbl, (p, G[p][g][0], e["lift"]), jaw=pb.jaw[p][g] if pb.close_fingers else None)
            ok_b = W.free([e["lift"], v["preins"]])
            t_wrs += perf_counter() - t0
            base = dict(ctx=attempt, part=p, installed=installed, table=table, x=x, g=int(g))
            segs.append(dict(base, kind="transit", a=np.asarray(start).tolist(), b=np.asarray(e["lift"]).tolist(),
                             wrs_free=bool(ok_a[0]), wrs_why=ok_a[1]))
            segs.append(dict(base, kind="transfer", a=np.asarray(e["lift"]).tolist(), b=np.asarray(v["preins"]).tolist(),
                             wrs_free=bool(ok_b[0]), wrs_why=ok_b[1]))
        done += 1
        print(f"ctx {attempt}: {p} k={k} pairs {scr['pairs']} -> {len(pick)} sampled; segments so far {len(segs)}", flush=True)
    W.set([], {})
    if not args.load:
        OUT.mkdir(exist_ok=True)
        seg_file.write_text(json.dumps(dict(segments=segs, wrs_s=t_wrs)), encoding="utf-8")
    n = len(segs)
    wrs = np.array([s["wrs_free"] for s in segs])
    print(f"\n{n} segments, WRS straight-line free {wrs.mean():.0%}; WRS check {1e3 * t_wrs / n:.0f} ms per segment\n")

    def ours(chk, spacing, radius, parts=("robot", "held", "extra")):
        """free flag per segment with the given checker / part cover; parts = components switched on."""
        cover = {p: surface_cover(sc.meshes[p], spacing, radius) for p in sc.ids}
        free = np.zeros(n, bool)
        torch.cuda.synchronize()
        t0 = perf_counter()
        by = {}
        for i, s in enumerate(segs):
            by.setdefault((s["ctx"], s["kind"]), []).append(i)
        n_states = 0
        for (ctx, kind), idx in by.items():
            s0 = segs[idx[0]]
            p = s0["part"]
            placed = [(f"{q}@{c}", q, sc.pose[q][c]) for q, c in s0["table"].items()]
            chk.set_world(chk.world_dict(s0["installed"], placed))
            A = np.array([segs[i]["a"] for i in idx])
            B = np.array([segs[i]["b"] for i in idx])
            n_states += int(np.sum(np.ceil(np.abs(B - A).max(1) / 0.02) + 1))
            if kind == "transit":
                xs = sorted({segs[i]["x"] for i in idx})
                ex = [chk.obstacle_spheres(cover[p], sc.pose[p][x]) for x in xs] if "extra" in parts else []
                ei = np.array([xs.index(segs[i]["x"]) for i in idx]) if ex else -1
                free[idx] = chk.batch_free([(A, B, -1, ei)], extra_bank=ex, step=0.02)[0]
            else:
                gs = sorted({segs[i]["g"] for i in idx})
                hb = [(cover[p], G[p][g][0]) for g in gs] if "held" in parts else []
                hi = np.array([gs.index(segs[i]["g"]) for i in idx]) if hb else -1
                jw = np.array([pb.jaw[p][segs[i]["g"]] for i in idx]) if pb.close_fingers else None
                free[idx] = chk.batch_free([(A, B, hi, -1, jw)], held_bank=hb, step=0.02)[0]
        torch.cuda.synchronize()
        return free, perf_counter() - t0, n_states

    def report(name, free, secs, n_states):
        fb = int(np.sum(~free & wrs))          # we block, WRS free
        miss = int(np.sum(free & ~wrs))        # we free, WRS blocks
        blocked = int(np.sum(~free))
        r = dict(name=name, agree=float(np.mean(free == wrs)), false_blocked=fb, missed=miss, ours_blocked=blocked,
                 false_share=fb / max(blocked, 1), ms=1e3 * secs, us_per_state=1e6 * secs / max(n_states, 1))
        tr = np.array([s["kind"] == "transit" for s in segs])
        r["false_transit"] = int(np.sum(~free & wrs & tr))
        r["false_transfer"] = int(np.sum(~free & wrs & ~tr))
        print(f"  {name:48s} agree {r['agree']:.0%} | false blocked {fb:4d} ({r['false_share']:.0%} of ours blocked; transit {r['false_transit']}, "
              f"transfer {r['false_transfer']}) | missed {miss:3d} | {r['ms']:.0f} ms, {r['us_per_state']:.2f} us/state", flush=True)
        return r

    results = []
    variants = [  # (density, buffer, spacing, radius)
        (1.0, 0.002, 0.006, 0.004),
        (1.0, 0.001, 0.006, 0.004),
        (1.0, 0.000, 0.006, 0.004),
        (1.0, 0.002, 0.003, 0.002),
        (1.0, 0.001, 0.003, 0.002),
        (1.0, 0.000, 0.003, 0.002),
        (1.0, 0.000, 0.002, 0.0015),
        (2.0, 0.002, 0.006, 0.004),
        (2.0, 0.001, 0.006, 0.004),
        (2.0, 0.000, 0.006, 0.004),
        (2.0, 0.001, 0.003, 0.002),
        (2.0, 0.000, 0.003, 0.002),
        (2.0, 0.000, 0.002, 0.0015),
    ]
    last = None
    for dens, buf, sp, rad in variants:
        if last != (dens, buf):
            chk = pb.chk if (dens, buf) == (pb.chk.density, pb.chk.buffer) else GpuChecker(sc, buffer=buf, density=dens)
            last = (dens, buf)
            ours(chk, sp, rad)                                                     # warm-up
            if (dens, buf, sp, rad) == variants[0]:
                print("ablation, current parameters (which component flags a WRS-free line):")
                for parts in (("robot",), ("robot", "held"), ("robot", "held", "extra")):
                    f, t, ns = ours(chk, sp, rad, parts)
                    results.append(report("current, components: " + "+".join(parts), f, t, ns))
                arm = chk.arm_spheres
                chk.arm_spheres = arm[:0]
                f, t, ns = ours(chk, sp, rad)
                results.append(report("current, without held-part-vs-arm check", f, t, ns))
                chk.arm_spheres = arm
                print("\nparameter variants:")
        f, t, ns = ours(chk, sp, rad)
        results.append(report(f"density {dens:g}, buffer {1e3 * buf:.0f} mm, cover {1e3 * sp:.0f}/{1e3 * rad:.0f} mm", f, t, ns))
        results[-1].update(density=dens, buffer=buf, spacing=sp, radius=rad)
    OUT.mkdir(exist_ok=True)
    (OUT / "diag_collision.json").write_text(json.dumps(dict(segments=n, wrs_free=float(wrs.mean()), wrs_ms=1e3 * t_wrs / n,
                                                              results=results), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
