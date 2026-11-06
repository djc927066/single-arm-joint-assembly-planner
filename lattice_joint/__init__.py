"""lattice_joint: exact joint search of assembly order x initial layout on the installed-set lattice.

Mainline since 2026-11-02 (the beam search joint_search stays as it is). See README.md.
    instance.py  scene, IK catalogue, stable orders, options, lazy geometry (from joint_search)
    lattice.py   steps 1-2: D(S, p) per arrow, step costs, forward DP F, root bound
    search.py    steps 3-5: reverse best-first, lazy pose checks, merging, incumbent U
    geometry.py  options and collision checks (unchanged from experiments/bls_all_orders_20261031)
    run.py       python -X utf8 -m lattice_joint.run --model triangular_frame6
"""
