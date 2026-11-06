"""Empirical optimality-gap + parallel-speedup harness for BSFS.

Runs the EXACT backward A*/branch-and-bound (discrete-domain global optimum) and
the ANYTIME beam on the SAME instance, and reports:

  * J_exact (optimal total end-effector path cost) vs J_beam,
  * optimality gap % = 100 * (J_beam - J_exact) / J_exact,
  * per-step certification counts and Hall-matching prunes,
  * full evaluate_layout (witness) call counts,
  * wall-clock for each,
  * optional parallel speedup vs worker count.

Because a finite-width beam has no optimality guarantee, this gap is exactly the
number the paper should report instead of claiming "near-optimal".

    python -m sealp.examples.layout.bsfs.bench_optimality_gap \\
        --asmdef sealp/assembly_sequence/_demo_output/yuanchair.asmdef \\
        --grasp-dir sealp/examples/grasp/yuanchair_grasp \\
        --goal-pos 0.373,0.0,0.0 --grid-spacing 0.06 --beam-width 6 \\
        --speedup 1,4,8
"""

from __future__ import annotations

import argparse
import copy
import time
from typing import List, Optional

from sealp.examples.layout.bsfs import run as bsfs_run


def _run_once(base_argv: List[str], mode: str, workers: Optional[int] = None):
    argv = list(base_argv) + ["--mode", mode]
    if workers is not None:
        argv += ["--workers", str(workers)]
    t0 = time.time()
    try:
        res = bsfs_run.main(argv)
    except SystemExit as e:
        print(f"[bench] {mode} produced no layout: {e}")
        res = None
    dt = time.time() - t0
    return res, dt


def main():
    ap = argparse.ArgumentParser(description="BSFS exact-vs-beam optimality gap")
    ap.add_argument("--asmdef", required=True)
    ap.add_argument("--grasp-dir", required=True)
    ap.add_argument("--goal-pos", default="0.373,0.0,0.0")
    ap.add_argument("--grid-spacing", default="0.06")
    ap.add_argument("--beam-width", default="6")
    ap.add_argument("--cand-per-part", default="10")
    ap.add_argument("--poses-per-xy", default="1")
    ap.add_argument("--all-centers", action="store_true")
    ap.add_argument("--center-spacing", default="0.06")
    ap.add_argument("--speedup", default="",
                    help="Comma worker counts to time the exact solver, e.g. 1,4,8")
    args = ap.parse_args()

    base = [
        "--asmdef", args.asmdef, "--grasp-dir", args.grasp_dir,
        "--goal-pos", args.goal_pos, "--grid-spacing", args.grid_spacing,
        "--beam-width", args.beam_width, "--cand-per-part", args.cand_per_part,
        "--poses-per-xy", args.poses_per_xy, "--workers", "1",
    ]
    if args.all_centers:
        base += ["--all-centers", "--center-spacing", args.center_spacing]

    print("\n########## EXACT (A*/branch-and-bound) ##########")
    exact, t_exact = _run_once(base, "exact")
    print("\n########## BEAM (anytime) ##########")
    beam, t_beam = _run_once(base, "beam")

    print("\n================ OPTIMALITY GAP ================")
    je = exact["total_cost"] if exact else None
    jb = beam["total_cost"] if beam else None
    print(f"J_exact = {je}")
    print(f"J_beam  = {jb}")
    if je and jb:
        gap = 100.0 * (jb - je) / je if je > 0 else float("nan")
        print(f"optimality gap = {gap:.2f}%")
    for tag, res, dt in (("exact", exact, t_exact), ("beam", beam, t_beam)):
        if res is None:
            continue
        st = res.get("stats", {})
        print(f"[{tag}] time={dt:.1f}s certifications={st.get('certifications')} "
              f"hall_pruned={st.get('hall_pruned')} "
              f"witness_calls={st.get('full_evaluate_layout')} "
              f"score={res.get('layout_score'):.4f}")

    speedup = [int(x) for x in str(args.speedup).split(",") if x.strip()]
    if speedup:
        print("\n================ PARALLEL SPEEDUP (exact) ================")
        t1 = None
        for w in speedup:
            _, dt = _run_once(base, "exact", workers=w)
            if t1 is None:
                t1 = dt
            print(f"workers={w:2d}  time={dt:6.1f}s  speedup={t1/dt:4.2f}x")


if __name__ == "__main__":
    main()
