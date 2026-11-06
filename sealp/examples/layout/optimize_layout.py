"""
Random Search Layout Optimization — End-to-End Demo
=====================================================

Run the ``RandomSearchOptimizer`` on the YuanChair assembly.

Pipeline:
1. Load assembly definition and generate grasps.
2. Run random search with configurable bounds.
3. Visualize the **best** layout in Panda3D with:
   - Robot at optimized base position
   - Parts at staging + ghost assembly poses
   - Feasibility indicators (green/red spheres)
4. Save convergence plot + best layout file.

Usage::

    python -m sealp.examples.layout.optimize_layout

Prerequisites:
    - Run ``python -m sealp.assembly_sequence.gen_yuanchair_asmdef``
      to generate the ``.asmdef`` file (if not already present).
"""

import os
import sys
import numpy as np

from wrs import wd, rm, mgm, mcm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from sealp.assembly_sequence import AssemblyDef
from sealp.layout import WorkspaceLayout, compute_metrics
from sealp.layout.optimizer_random import RandomSearchOptimizer


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════
def build_grasp_cache(assembly_def):
    from sealp.examples.grasp.planning import plan_grasps
    cache = {}
    for alias, path in assembly_def.models.items():
        if alias in cache:
            continue
        print(f"  Planning grasps for '{alias}' ...")
        obj = mcm.CollisionModel(initor=path)
        gc, _ = plan_grasps(obj, max_samples=50)
        cache[alias] = gc
        print(f"    → {len(gc)} grasps")
    return cache


def plot_convergence(history, out_dir):
    """Plot optimization score convergence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = [s for s, _ in history]
    n = len(scores)

    # Cumulative best
    cum_best = []
    best_so_far = -np.inf
    for s in scores:
        best_so_far = max(best_so_far, s)
        cum_best.append(best_so_far)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(range(n), scores, s=12, alpha=0.4, color="#45B7D1",
               label="Sample score", zorder=2)
    ax.plot(range(n), cum_best, color="#E74C3C", linewidth=2.0,
            label="Best so far", zorder=3)
    ax.fill_between(range(n), 0, cum_best, color="#E74C3C",
                     alpha=0.07, zorder=1)

    ax.set_xlabel("Sample Index", fontsize=11)
    ax.set_ylabel("Composite Score", fontsize=11)
    ax.set_title("Random Search Optimizer — Convergence",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(out_dir, "optimization_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Convergence plot saved: {path}")
    plt.close(fig)


def plot_robot_positions(history, best_layout, out_dir):
    """Plot sampled robot base positions (top view)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = np.array([s for s, _ in history])
    positions = np.array([l.robot_base_pos[:2] for _, l in history])

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=scores, cmap="RdYlGn", s=25, alpha=0.6,
        edgecolors="white", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="Composite Score", shrink=0.8)

    # Best position
    bp = best_layout.robot_base_pos[:2]
    ax.scatter([bp[0]], [bp[1]], marker="*", s=300, c="red",
               edgecolors="black", linewidths=1.2, zorder=5,
               label=f"Best ({bp[0]:.3f}, {bp[1]:.3f})")

    # Assembly station
    ax.scatter([0], [0], marker="s", s=100, c="blue", alpha=0.5,
               label="Assembly Station")

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title("Sampled Robot Base Positions (Top View)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(out_dir, "optimization_positions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Position plot saved: {path}")
    plt.close(fig)


def visualize_best_layout(base, layout, assembly_def, report):
    """Show feasibility indicators on the best layout."""
    RADIUS = 0.012
    world_poses = assembly_def.compute_world_poses(
        fixture_pos=layout.assembly_station_pos,
        fixture_rotmat=layout.assembly_station_rotmat)

    for sf in report.steps:
        pid = sf.part_id
        staging = layout.get_staging(pid)
        if staging is not None:
            pick_pos, _ = staging
            color = np.array([0.15, 0.85, 0.15, 0.85]) if sf.pick_collision_free \
                else np.array([0.90, 0.15, 0.15, 0.85])
            s = mcm.gen_sphere(radius=RADIUS)
            s.pos = pick_pos + np.array([0, 0, 0.04])
            s.rgba = color
            s.attach_to(base)
        if pid in world_poses:
            place_pos, _ = world_poses[pid]
            color = np.array([0.15, 0.85, 0.15, 0.85]) if sf.place_collision_free \
                else np.array([0.90, 0.15, 0.15, 0.85])
            s = mcm.gen_sphere(radius=RADIUS)
            s.pos = place_pos + np.array([0, 0, 0.04])
            s.rgba = color
            s.attach_to(base)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def main():
    import wrs.robot_sim.robots.piper.piper_single_arm as psa

    # ------------------------------------------------------------------
    # 1. Scene setup
    # ------------------------------------------------------------------
    base = wd.World(cam_pos=[1.2, 0.8, 1.0], lookat_pos=[0.25, 0, 0.05])
    mgm.gen_frame(ax_length=0.15).attach_to(base)

    # Ground
    ground = mcm.gen_box(
        xyz_lengths=rm.vec(2, 2, 0.01),
        rgb=rm.vec(0.75, 0.75, 0.75), alpha=1)
    ground.pos = np.array([0.3, 0, -0.005])
    ground.attach_to(base)

    # ------------------------------------------------------------------
    # 2. Load assembly definition
    # ------------------------------------------------------------------
    asmdef_dir = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "assembly_sequence", "_demo_output")
    asmdef_path = os.path.abspath(
        os.path.join(asmdef_dir, "yuanchair.asmdef"))

    if not os.path.isfile(asmdef_path):
        print("Assembly definition not found, generating...")
        from sealp.assembly_sequence.gen_yuanchair_asmdef import main as gen_asm
        gen_asm()

    asm = AssemblyDef.load(asmdef_path)
    print(f"Loaded: {asm.name} ({asm.n_parts} parts, {asm.n_steps} steps)")

    # ------------------------------------------------------------------
    # 3. Robot + grasps
    # ------------------------------------------------------------------
    robot = psa.PiperSglArm(enable_cc=True)

    print("\nPlanning grasps...")
    grasp_cache = build_grasp_cache(asm)

    # ------------------------------------------------------------------
    # 4. Seed layout (staging positions)
    # ------------------------------------------------------------------
    seed_layout = WorkspaceLayout(
        robot_base_pos=np.array([0.0, 0.0, 0.0]),
        staging_positions={
            "seat":   (np.array([0.30,  0.00, 0.00]), np.eye(3)),
            "leg_fl": (np.array([0.10,  0.20, 0.00]), np.eye(3)),
            "leg_fr": (np.array([0.05, -0.24, 0.00]), np.eye(3)),
            "leg_bl": (np.array([0.25,  0.24, 0.00]), np.eye(3)),
            "leg_br": (np.array([0.25, -0.24, 0.00]), np.eye(3)),
        },
        name="seed_layout",
    )

    # ------------------------------------------------------------------
    # 5. Run optimizer
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Random Search Optimization")
    print("=" * 60)

    optimizer = RandomSearchOptimizer(
        n_samples=50,
        robot_bounds={
            "xy_min": [-0.15, -0.20],
            "xy_max": [0.20,  0.20],
            "z_range": [0.0, 0.0],
            "yaw_range": [-0.5, 0.5],
        },
        randomize_staging=False,
        seed=42,
        max_grasps_per_step=15,
        verbose=True,
    )

    result = optimizer.optimize(
        assembly_def=asm,
        robot=robot,
        grasp_cache=grasp_cache,
        obstacle_list=[ground],
        seed_layout=seed_layout,
    )

    print(f"\n{result.summary()}")

    # ------------------------------------------------------------------
    # 6. Output directory
    # ------------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "_output")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 7. Save convergence + position plots
    # ------------------------------------------------------------------
    if result.history:
        print("\nGenerating plots...")
        plot_convergence(result.history, out_dir)
        if result.best_layout is not None:
            plot_robot_positions(result.history, result.best_layout, out_dir)

    # ------------------------------------------------------------------
    # 8. Save best layout
    # ------------------------------------------------------------------
    if result.best_layout is not None:
        layout_path = os.path.join(out_dir, "best_layout.layout")
        result.best_layout.save(layout_path)
        print(f"\nBest layout saved: {layout_path}")

        # Metrics for best layout
        if result.best_report is not None:
            metrics = compute_metrics(result.best_layout, result.best_report)
            print(f"\n{metrics.summary()}")

    # ------------------------------------------------------------------
    # 9. Visualize best layout in Panda3D
    # ------------------------------------------------------------------
    best = result.best_layout
    if best is not None:
        print(f"\nVisualizing best layout: {best.name}")
        print(f"  Robot base: {best.robot_base_pos.tolist()}")

        # Show robot at optimized position
        robot_vis = psa.PiperSglArm(
            pos=best.robot_base_pos,
            rotmat=best.robot_base_rotmat,
            enable_cc=False)
        robot_vis.gen_meshmodel(alpha=0.35).attach_to(base)

        # Show staging parts
        staging_colors = {
            "seat":   np.array([0.9, 0.6, 0.3, 0.7]),
            "leg_fl": np.array([0.3, 0.7, 0.3, 0.7]),
            "leg_fr": np.array([0.3, 0.3, 0.8, 0.7]),
            "leg_bl": np.array([0.8, 0.3, 0.3, 0.7]),
            "leg_br": np.array([0.7, 0.3, 0.7, 0.7]),
        }
        for pid in asm.part_ids:
            st = best.get_staging(pid)
            if st is None:
                continue
            pos, rotmat = st
            mp = asm.model_path(pid)
            if os.path.isfile(mp):
                m = mcm.CollisionModel(initor=mp)
                m.pos = pos
                m.rotmat = rotmat
                m.rgba = staging_colors.get(pid, np.array([0.5, 0.5, 0.5, 0.7]))
                m.attach_to(base)
                mgm.gen_frame(pos=pos, ax_length=0.025).attach_to(base)

        # Show assembly ghosts
        world_poses = asm.compute_world_poses()
        for pid in asm.part_ids:
            if pid not in world_poses:
                continue
            gp, gr = world_poses[pid]
            mp = asm.model_path(pid)
            if os.path.isfile(mp):
                ghost = mcm.CollisionModel(initor=mp)
                ghost.pos = gp
                ghost.rotmat = gr
                ghost.alpha = 0.15
                ghost.attach_to(base)

        # Show feasibility indicators
        if result.best_report is not None:
            visualize_best_layout(base, best, asm, result.best_report)

    base.run()


if __name__ == "__main__":
    main()
