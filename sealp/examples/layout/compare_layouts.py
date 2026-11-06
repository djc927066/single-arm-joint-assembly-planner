"""
Layout Comparison — Matplotlib Charts
========================================

Evaluate several candidate layouts for the YuanChair assembly and
produce side-by-side bar charts comparing:

  - Feasibility rate (% of steps that are kinematically feasible)
  - Average manipulability
  - Composite score
  - Per-step breakdown (heatmap)

The evaluation is headless — no Panda3D window is opened.
Results are saved as ``.png`` charts.

Usage::

    python -m sealp.examples.layout.compare_layouts

Prerequisites:
    - Run ``python -m sealp.assembly_sequence.gen_yuanchair_asmdef``
      to generate the ``.asmdef`` file (if not already present).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import wrs.modeling.collision_model as mcm

from sealp.assembly_sequence import AssemblyDef
from sealp.layout import WorkspaceLayout, evaluate_layout, compute_metrics


# ══════════════════════════════════════════════════════════════
#  Helper — plan grasps (headless)
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


# ══════════════════════════════════════════════════════════════
#  Define candidate layouts
# ══════════════════════════════════════════════════════════════
def define_candidate_layouts():
    """Return a list of named WorkspaceLayouts to compare."""
    staging_base = {
        "seat":   (np.array([0.30,  0.00, 0.00]), np.eye(3)),
        "leg_fl": (np.array([0.10,  0.20, 0.00]), np.eye(3)),
        "leg_fr": (np.array([0.05, -0.24, 0.00]), np.eye(3)),
        "leg_bl": (np.array([0.25,  0.24, 0.00]), np.eye(3)),
        "leg_br": (np.array([0.25, -0.24, 0.00]), np.eye(3)),
    }

    layouts = []

    # Layout A: robot at origin (default)
    la = WorkspaceLayout(
        robot_base_pos=np.array([0.0, 0.0, 0.0]),
        staging_positions=dict(staging_base),
        name="A: Origin",
    )
    layouts.append(la)

    # Layout B: robot shifted right
    lb = WorkspaceLayout(
        robot_base_pos=np.array([0.15, 0.0, 0.0]),
        staging_positions=dict(staging_base),
        name="B: Shifted +X",
    )
    layouts.append(lb)

    # Layout C: robot shifted left
    lc = WorkspaceLayout(
        robot_base_pos=np.array([-0.10, 0.0, 0.0]),
        staging_positions=dict(staging_base),
        name="C: Shifted -X",
    )
    layouts.append(lc)

    # Layout D: robot rotated 30° about z
    yaw = np.radians(30)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotmat30 = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.]])
    ld = WorkspaceLayout(
        robot_base_pos=np.array([0.0, -0.05, 0.0]),
        robot_base_rotmat=rotmat30,
        staging_positions=dict(staging_base),
        name="D: Rotated 30°",
    )
    layouts.append(ld)

    # Layout E: robot far from parts (should score poorly)
    le = WorkspaceLayout(
        robot_base_pos=np.array([-0.3, 0.0, 0.0]),
        staging_positions=dict(staging_base),
        name="E: Far Away",
    )
    layouts.append(le)

    return layouts


# ══════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════
def plot_comparison(names, metrics_list, reports, out_dir):
    """Generate comparison bar charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(names)
    x = np.arange(n)
    bar_w = 0.55

    # ── Color palette ────────────────────────────────────────
    colors = ["#4ECDC4", "#45B7D1", "#F7DC6F", "#F1948A", "#BB8FCE"]

    # ------------------------------------------------------------------
    # Figure 1: Summary metrics (3 subplots)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Layout Comparison — YuanChair Assembly",
                 fontsize=15, fontweight="bold", y=1.02)

    # -- Feasibility rate --
    ax = axes[0]
    vals = [m.feasibility_rate * 100 for m in metrics_list]
    bars = ax.bar(x, vals, width=bar_w, color=colors[:n], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Feasibility Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Feasibility Rate")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2,
                f"{v:.0f}%", ha="center", fontsize=9)
    ax.axhline(100, color="green", linestyle="--", alpha=0.3, linewidth=1)

    # -- Average manipulability --
    ax = axes[1]
    vals = [m.manipulability_avg for m in metrics_list]
    bars = ax.bar(x, vals, width=bar_w, color=colors[:n], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Avg Manipulability")
    ax.set_title("Average Manipulability")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.03,
                f"{v:.4f}", ha="center", fontsize=8)

    # -- Composite score --
    ax = axes[2]
    vals = [m.composite_score for m in metrics_list]
    bars = ax.bar(x, vals, width=bar_w, color=colors[:n], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Composite Score")
    ax.set_title("Composite Score (higher = better)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.03,
                f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    path1 = os.path.join(out_dir, "layout_comparison_summary.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path1}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Per-step feasibility heatmap
    # ------------------------------------------------------------------
    if not reports:
        return

    step_ids = [sf.part_id for sf in reports[0].steps]
    n_steps = len(step_ids)
    data = np.zeros((n, n_steps))

    for i, rpt in enumerate(reports):
        for j, sf in enumerate(rpt.steps):
            if sf.feasible:
                data[i, j] = 1.0
            elif sf.pick_reachable or sf.place_reachable:
                data[i, j] = 0.5  # partially reachable
            else:
                data[i, j] = 0.0

    fig2, ax2 = plt.subplots(figsize=(max(8, n_steps * 1.2), max(4, n * 0.8)))
    cmap = plt.cm.RdYlGn
    im = ax2.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax2.set_xticks(np.arange(n_steps))
    ax2.set_xticklabels(step_ids, fontsize=9, rotation=30, ha="right")
    ax2.set_yticks(np.arange(n))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("Assembly Step (part)")
    ax2.set_ylabel("Layout")
    ax2.set_title("Per-Step Feasibility Heatmap\n"
                   "(Green = feasible, Yellow = partial, Red = infeasible)",
                   fontsize=12, fontweight="bold")

    # Annotate cells
    for i in range(n):
        for j in range(n_steps):
            label = "✓" if data[i, j] == 1.0 else ("~" if data[i, j] == 0.5 else "✗")
            color = "white" if data[i, j] < 0.4 else "black"
            ax2.text(j, i, label, ha="center", va="center",
                     fontsize=12, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax2, shrink=0.7, label="Feasibility")
    plt.tight_layout()
    path2 = os.path.join(out_dir, "layout_comparison_heatmap.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path2}")
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Figure 3: Manipulability per step (grouped bar)
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(max(8, n_steps * 1.5), 5))
    bar_w2 = 0.8 / n
    for i, (name, rpt) in enumerate(zip(names, reports)):
        manip_vals = [sf.manipulability_avg for sf in rpt.steps]
        offsets = x_steps + i * bar_w2 - (n - 1) * bar_w2 / 2 \
            if 'x_steps' in dir() else np.arange(n_steps) + i * bar_w2 - (n - 1) * bar_w2 / 2
        ax3.bar(np.arange(n_steps) + i * bar_w2 - (n - 1) * bar_w2 / 2,
                manip_vals, width=bar_w2, label=name,
                color=colors[i % len(colors)], edgecolor="white", alpha=0.85)

    ax3.set_xticks(np.arange(n_steps))
    ax3.set_xticklabels(step_ids, fontsize=9, rotation=30, ha="right")
    ax3.set_xlabel("Assembly Step (part)")
    ax3.set_ylabel("Manipulability")
    ax3.set_title("Per-Step Manipulability by Layout", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    path3 = os.path.join(out_dir, "layout_comparison_manipulability.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path3}")
    plt.close(fig3)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def main():
    import wrs.robot_sim.robots.piper.piper_single_arm as psa

    # ------------------------------------------------------------------
    # 1. Load assembly definition
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
    # 2. Ground obstacle
    # ------------------------------------------------------------------
    ground = mcm.gen_box(
        xyz_lengths=np.array([2.0, 2.0, 0.01]),
        rgb=np.array([0.75, 0.75, 0.75]), alpha=1)
    ground.pos = np.array([0.3, 0, -0.005])

    # ------------------------------------------------------------------
    # 3. Robot (shared across evaluations — repositioned each time)
    # ------------------------------------------------------------------
    robot = psa.PiperSglArm(enable_cc=True)

    # ------------------------------------------------------------------
    # 4. Plan grasps once
    # ------------------------------------------------------------------
    print("\nPlanning grasps (one time) ...\n")
    grasp_cache = build_grasp_cache(asm)

    # ------------------------------------------------------------------
    # 5. Define and evaluate candidate layouts
    # ------------------------------------------------------------------
    layouts = define_candidate_layouts()
    names = []
    metrics_list = []
    reports = []

    for layout in layouts:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {layout.name}")
        print(f"  Robot at: {layout.robot_base_pos.tolist()}")
        print(f"{'=' * 60}")

        report = evaluate_layout(
            layout=layout,
            assembly_def=asm,
            robot=robot,
            grasp_cache=grasp_cache,
            obstacle_list=[ground],
            max_grasps_per_step=20,
            verbose=True,
        )
        metrics = compute_metrics(layout, report)

        names.append(layout.name)
        metrics_list.append(metrics)
        reports.append(report)

        print(f"\n{metrics.summary()}")

    # ------------------------------------------------------------------
    # 6. Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 75}")
    print(f"{'Layout':<20} {'Feas%':>7} {'ManipAvg':>10} {'ManipMin':>10} {'Score':>8}")
    print(f"{'-' * 75}")
    for name, m in zip(names, metrics_list):
        print(f"{name:<20} {m.feasibility_rate:>6.0%} "
              f"{m.manipulability_avg:>10.4f} "
              f"{m.manipulability_min:>10.4f} "
              f"{m.composite_score:>8.4f}")
    print(f"{'=' * 75}")

    # ------------------------------------------------------------------
    # 7. Generate charts
    # ------------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "_output")
    os.makedirs(out_dir, exist_ok=True)
    print("\nGenerating comparison charts...")
    plot_comparison(names, metrics_list, reports, out_dir)

    print("\nDone! Charts saved to:", out_dir)


if __name__ == "__main__":
    main()
