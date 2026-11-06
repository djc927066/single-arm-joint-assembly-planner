"""Visualise the completed CrossRailFrameV1 assembly.

The script reads the real STL meshes and final relative transforms from the
``.asmdef`` file.  It produces a two-view PNG by default.  Pass ``--show`` to
open an interactive Matplotlib window that can be rotated with the mouse.

Examples
--------
Save the completed assembly image::

    python -m sealp.examples.visualization.visualize_cross_rail_frame

Open an interactive window::

    python -m sealp.examples.visualization.visualize_cross_rail_frame --show

Show only the assembly state after step 3::

    python -m sealp.examples.visualization.visualize_cross_rail_frame --step 3 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_ASMDEF = (
    PROJECT_ROOT
    / "sealp"
    / "assembly_sequence"
    / "_demo_output"
    / "cross_rail_frame_v1.asmdef"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "sealp"
    / "examples"
    / "visualization"
    / "_output"
    / "cross_rail_frame_completed.png"
)

PART_STYLE = {
    "base_plate": ("#59636e", "Base plate (fixed)"),
    "post_l": ("#e9a23b", "Left post"),
    "post_r": ("#f2c14e", "Right post"),
    "rail_l": ("#2a9d8f", "Left rail"),
    "rail_r": ("#56b4a9", "Right rail"),
    "cap_plate": ("#5875d5", "Cap plate"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the CrossRailFrameV1 final assembly from its asmdef."
    )
    parser.add_argument(
        "--asmdef",
        type=Path,
        default=DEFAULT_ASMDEF,
        help="CrossRailFrameV1 .asmdef file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG file written by the renderer.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        choices=range(0, 6),
        metavar="0..5",
        help="Last visible assembly step; 5 is the completed assembly.",
    )
    parser.add_argument(
        "--explode",
        type=float,
        default=0.0,
        help="Exploded-view distance in metres (try 0.025).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a mouse-rotatable Matplotlib window after saving the PNG.",
    )
    return parser.parse_args()


def _transform_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform


def _exploded_offset(part_id: str, distance: float) -> np.ndarray:
    """Return a readable exploded-view offset without changing the asmdef."""
    directions = {
        "base_plate": np.array([0.0, 0.0, 0.0]),
        "post_l": np.array([-0.35, 0.0, 0.45]),
        "post_r": np.array([0.35, 0.0, 0.45]),
        "rail_l": np.array([-1.0, 0.0, 0.15]),
        "rail_r": np.array([1.0, 0.0, 0.15]),
        "cap_plate": np.array([0.0, 0.0, 1.0]),
    }
    return directions[part_id] * float(distance)


def _load_assembly(asmdef_path: Path, last_step: int, explode: float):
    import trimesh

    from sealp.assembly_sequence.asmdef import AssemblyDef

    asmdef_path = asmdef_path.resolve()
    if not asmdef_path.is_file():
        raise FileNotFoundError(f"Assembly definition not found: {asmdef_path}")

    assembly = AssemblyDef.load(asmdef_path)
    world_poses = assembly.compute_world_poses()
    step_by_part = {step.part_id: step.step_id for step in assembly.steps}

    meshes: Dict[str, object] = {}
    poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for part_id, part in assembly.parts.items():
        if step_by_part[part_id] > last_step:
            continue
        model_path = Path(assembly.models[part.model])
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Mesh for {part_id!r} was not found: {model_path}"
            )
        mesh = trimesh.load_mesh(model_path, force="mesh", process=False)
        position, rotation = world_poses[part_id]
        position = position + _exploded_offset(part_id, explode)
        mesh.apply_transform(_transform_matrix(position, rotation))
        meshes[part_id] = mesh
        poses[part_id] = (position, rotation)
    return assembly, meshes, poses


def _set_equal_3d_axes(ax, bounds: np.ndarray) -> None:
    center = bounds.mean(axis=0)
    span = max(float(np.max(bounds[1] - bounds[0])), 0.16)
    radius = span * 0.58
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(min(-0.006, center[2] - radius * 0.35), center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 0.82))


def _draw_direction_arrows(ax, assembly, poses, scale: float = 1.0) -> None:
    """Draw the final insertion directions stored in the assembly definition."""
    arrow_len = 0.030 * scale
    # StepDef.insertion_axis is explicitly stored in world coordinates.
    arrows = {
        step.part_id: np.asarray(step.insertion_axis, dtype=float)
        for step in assembly.steps
        if step.insertion_axis is not None
    }
    for part_id, direction in arrows.items():
        if part_id not in poses:
            continue
        position, _rotation = poses[part_id]
        world_direction = direction / np.linalg.norm(direction)
        start = position - world_direction * arrow_len
        ax.quiver(
            *start,
            *world_direction,
            length=arrow_len,
            normalize=True,
            color="#c93c37",
            linewidth=1.7,
            arrow_length_ratio=0.22,
        )


def _render_axis(ax, assembly, meshes, poses, bounds, *, elev: float, azim: float,
                 title: str, show_arrows: bool) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for part_id, mesh in meshes.items():
        color, _ = PART_STYLE[part_id]
        triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
        collection = Poly3DCollection(
            triangles,
            facecolor=color,
            edgecolor="#263238",
            linewidth=0.28,
            alpha=0.94,
        )
        ax.add_collection3d(collection)

    # A translucent reference table makes the scale and the fixed base obvious.
    table_z = -0.004
    table = np.array([
        [-0.12, -0.105, table_z],
        [0.12, -0.105, table_z],
        [0.12, 0.105, table_z],
        [-0.12, 0.105, table_z],
    ])
    ax.add_collection3d(Poly3DCollection(
        [table], facecolor="#dce3e7", edgecolor="#aab4ba",
        linewidth=0.5, alpha=0.30,
    ))

    if show_arrows:
        _draw_direction_arrows(ax, assembly, poses)

    _set_equal_3d_axes(ax, bounds)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("X / m", labelpad=5)
    ax.set_ylabel("Y / m", labelpad=5)
    ax.set_zlabel("Z / m", labelpad=5)
    ax.grid(True, alpha=0.22)


def render(asmdef_path: Path, output_path: Path, last_step: int,
           explode: float, show: bool) -> Path:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    assembly, meshes, poses = _load_assembly(
        asmdef_path=asmdef_path,
        last_step=last_step,
        explode=explode,
    )
    if not meshes:
        raise RuntimeError("No mesh was selected for rendering.")

    all_bounds = np.asarray([mesh.bounds for mesh in meshes.values()])
    bounds = np.array([all_bounds[:, 0].min(axis=0),
                       all_bounds[:, 1].max(axis=0)])

    figure = plt.figure(figsize=(13.2, 7.0), facecolor="#f7f8fa")
    ax_front = figure.add_subplot(1, 2, 1, projection="3d")
    ax_side = figure.add_subplot(1, 2, 2, projection="3d")
    state_name = "Completed assembly" if last_step == 5 else f"After step {last_step}"
    _render_axis(
        ax_front, assembly, meshes, poses, bounds,
        elev=23, azim=-58, title=f"{state_name} - perspective",
        show_arrows=True,
    )
    _render_axis(
        ax_side, assembly, meshes, poses, bounds,
        elev=16, azim=2, title="Side view",
        show_arrows=False,
    )

    handles: Iterable[object] = [
        Patch(facecolor=PART_STYLE[pid][0], edgecolor="#263238",
              label=PART_STYLE[pid][1])
        for pid in meshes
    ]
    handles = list(handles) + [
        Line2D([0], [0], color="#c93c37", lw=2,
               label="Final insertion direction")
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
        fontsize=9,
    )
    figure.suptitle(
        f"{assembly.name} - final target geometry",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5, 0.932,
        "6 parts: fixed base, 2 posts, 2 side-inserted rails and a top cap",
        ha="center", va="center", fontsize=10, color="#455a64",
    )
    figure.subplots_adjust(left=0.02, right=0.98, top=0.865, bottom=0.16,
                           wspace=0.02)

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight",
                   facecolor=figure.get_facecolor())
    print(f"Rendered {len(meshes)} parts from {asmdef_path.resolve()}")
    print(f"Saved image: {output_path}")
    if show:
        print("Close the Matplotlib window to exit.")
        plt.show()
    else:
        plt.close(figure)
    return output_path


def main() -> None:
    args = _parse_args()
    render(
        asmdef_path=args.asmdef,
        output_path=args.output,
        last_step=args.step,
        explode=args.explode,
        show=args.show,
    )


if __name__ == "__main__":
    main()
