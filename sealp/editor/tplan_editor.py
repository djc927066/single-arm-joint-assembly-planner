"""
Task Plan Editor — v1.0
========================

Interactive Panda3D-based editor for configuring ``.tplan`` files.
Load an ``.asmdef``, visually arrange staging positions, robot base,
and fixture location on a table surface, then save as ``.tplan``.

Features:
- Configurable table height
- Robot mesh visualization (extensible via robot_factory)
- FSReferencePoses for stable part placement on table
- Staging parts (solid, draggable on table)
- Assembly ghosts (transparent, fixture-relative)
- Fixture marker (draggable, moves all ghosts)
- Per-step primitive type selection
- Selective grasp visualization (stub)

Run::

    python -m sealp.editor.run_tplan_editor
    python -m sealp.editor.run_tplan_editor chair.asmdef
    python -m sealp.editor.run_tplan_editor chair_plan.tplan
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import numpy as np
from pathlib import Path

# TracIK must be imported BEFORE panda3d to avoid native library conflicts
try:
    from trac_ik import TracIK as _TracIK  # noqa: F401
except ImportError:
    pass

from panda3d.core import (
    TextNode, LPoint3f, LVecBase3f, LVecBase4f,
    Point3, Vec3, Vec4, Plane, LPoint3,
    WindowProperties,
)
from direct.gui.DirectGui import (
    DirectFrame, DirectLabel, DirectButton,
    DirectScrolledFrame, DirectEntry, DGG,
    DirectOptionMenu,
)

import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.basis.robot_math as rm

from sealp.assembly_sequence import (
    AssemblyDef, TaskPlan, StepParams, StagingPose, RobotConfig, Primitive,
)
from sealp.editor.editor_gui import (
    create_panel, create_section_header, create_label,
    create_entry, create_button, create_accent_button,
    create_list_item, create_separator, ConsoleWindow,
    BG_DARK, BG_MID, BG_LIGHT, ACCENT, ACCENT_BRIGHT, ACCENT_DIM,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_ACCENT, TEXT_OK, TEXT_WARN,
    TEXT_SIZE, SMALL_TEXT, TITLE_SIZE, SECTION_SIZE,
    LEFT_MARGIN, TOP_MARGIN, BOTTOM_MARGIN, LIST_BG,
)
from sealp.editor.transform_handler import (
    TransformHandler, TransformMode, AxisConstraint,
)
from sealp.editor.robot_factory import create_robot, available_robots

# ── Layout constants ─────────────────────────────────────────
RIGHT_W = 0.30
LEFT_W = 0.28
CONSOLE_H = 0.22
STATUS_H = 0.040
BTN_W = 0.17
BTN_GAP = 0.03

# ── Part colors ──────────────────────────────────────────────
STAGING_COLORS = [
    np.array([0.40, 0.76, 0.96, 1.0]),
    np.array([0.96, 0.65, 0.35, 1.0]),
    np.array([0.55, 0.85, 0.50, 1.0]),
    np.array([0.95, 0.50, 0.50, 1.0]),
    np.array([0.75, 0.60, 0.88, 1.0]),
    np.array([0.98, 0.85, 0.40, 1.0]),
    np.array([0.55, 0.80, 0.78, 1.0]),
    np.array([0.90, 0.55, 0.75, 1.0]),
]
GHOST_ALPHA = 0.25
SELECTED_COLOR = np.array([1.0, 1.0, 0.3, 1.0])
FIXTURE_COLOR = np.array([0.2, 0.9, 0.4, 1.0])
TABLE_COLOR = np.array([0.42, 0.40, 0.38])


class TplanEditor:
    """Task Plan Editor — configure .tplan files interactively."""

    def __init__(self, input_file: str = None):
        # ── Create world ─────────────────────────────────────
        self.world = wd.World(
            cam_pos=np.array([1.2, -1.0, 0.9]),
            lookat_pos=np.array([0.2, 0, 0.1]),
            w=1600, h=900,
        )
        self.world.setBackgroundColor(0.20, 0.20, 0.22, 1)

        props = WindowProperties()
        props.setTitle("SEALP Task Plan Editor")
        self.world.win.requestProperties(props)

        # ── State ────────────────────────────────────────────
        self._asmdef: AssemblyDef | None = None
        self._tplan: TaskPlan = TaskPlan(name="New Task Plan")
        self._current_file: str | None = None
        self._table_height: float = 0.0
        self._table_size: tuple = (0.8, 0.6)  # x, y extents

        # Scene objects
        self._robot = None          # robot sim instance
        self._robot_mesh = None     # attached mesh model
        self._table_model = None    # table collision model
        self._fixture_marker = None # fixture position marker
        self._staging_models: dict[str, mcm.CollisionModel] = {}
        self._staging_colors: dict[str, np.ndarray] = {}
        self._ghost_models: dict[str, mcm.CollisionModel] = {}
        self._fs_ref_poses: dict[str, list] = {}  # model_path → poses

        # Selection
        self._selected_type: str | None = None   # "robot", "fixture", part_id
        self._selected_id: str | None = None
        self._color_idx = 0
        self._selection_wireframe = None  # wireframe overlay NodePath

        # Placement dropdown state
        self._placement_menu = None       # DirectOptionMenu widget
        self._placement_label = None      # "Placement:" label
        self._current_placement_idx: dict[str, int] = {}  # part_id → idx

        # Transform
        self.transform = TransformHandler(on_mode_change=self._on_mode_change)
        self._grab_plane_z: float = 0.0
        self._grab_start_mouse: tuple | None = None

        # ── Build scene ──────────────────────────────────────
        self._draw_table()
        self._draw_ground_grid()

        # ── Build GUI ────────────────────────────────────────
        self._build_right_panel()
        self._build_left_panel()
        self._build_status_bar()
        self._build_console()

        # ── Keys ─────────────────────────────────────────────
        self._setup_keys()

        # ── Per-frame ────────────────────────────────────────
        self.world.taskMgr.add(self._editor_update, "tplan_editor_update")

        # ── Deferred load (after event loop starts) ───────────
        self._pending_load = None
        if input_file and os.path.isfile(input_file):
            self._pending_load = input_file
        else:
            self.console.log_info(
                "Ready.  Load an .asmdef or .tplan file to begin.")
        self.world.taskMgr.doMethodLater(
            0.5, self._deferred_load, "deferred_load")

    def _deferred_load(self, task):
        """Load file after the event loop has started to avoid segfaults."""
        if self._pending_load:
            filepath = self._pending_load
            self._pending_load = None
            ext = os.path.splitext(filepath)[1].lower()
            try:
                print(f"[TplanEditor] Loading: {filepath}")
                if ext == ".tplan":
                    self._do_load_tplan(filepath)
                elif ext == ".asmdef":
                    self._do_load_asmdef(filepath)
                else:
                    self.console.log_warn(f"Unknown file type: {ext}")
                print("[TplanEditor] Load complete.")
            except Exception as e:
                print(f"[TplanEditor] Load error: {e}")
                self.console.log_warn(f"Load error: {e}")
                traceback.print_exc()
        return task.done

    # ==============================================================
    # Scene: table, grid, fixture
    # ==============================================================
    def _draw_table(self):
        """Draw the table surface."""
        if self._table_model:
            self._table_model.detach()
        sx, sy = self._table_size
        self._table_model = mcm.gen_box(
            xyz_lengths=np.array([sx, sy, 0.02]),
            pos=np.array([sx / 2 - 0.1, 0, self._table_height - 0.01]),
            rgb=TABLE_COLOR, alpha=0.85,
        )
        self._table_model.attach_to(self.world)

    def _draw_ground_grid(self, size=0.8, step=0.1):
        n = int(size / step)
        for i in range(-n, n + 1):
            v = i * step
            z = self._table_height
            rgb = np.array([0.32, 0.32, 0.34])
            alpha = 0.3
            if i == 0:
                rgb = np.array([0.45, 0.45, 0.48])
                alpha = 0.5
            mgm.gen_stick(
                spos=np.array([v, -size, z]),
                epos=np.array([v, size, z]),
                radius=0.0004, rgb=rgb, alpha=alpha,
            ).attach_to(self.world)
            mgm.gen_stick(
                spos=np.array([-size, v, z]),
                epos=np.array([size, v, z]),
                radius=0.0004, rgb=rgb, alpha=alpha,
            ).attach_to(self.world)

    def _draw_fixture_marker(self):
        """Draw or update the fixture position marker (an arrow)."""
        if self._fixture_marker:
            self._fixture_marker.detach()
        pos = self._tplan.fixture_pos.copy()
        pos[2] = self._table_height
        self._fixture_marker = mgm.gen_frame(
            pos=pos, rotmat=self._tplan.fixture_rotmat,
            ax_length=0.08, ax_radius=0.003,
        )
        self._fixture_marker.attach_to(self.world)
        # Also draw a small marker box
        marker = mcm.gen_box(
            xyz_lengths=np.array([0.03, 0.03, 0.005]),
            pos=pos + np.array([0, 0, 0.002]),
            rgb=FIXTURE_COLOR[:3], alpha=0.8,
        )
        marker.attach_to(self.world)

    # ==============================================================
    # Robot
    # ==============================================================
    def _spawn_robot(self):
        """Create and display the robot at its base position."""
        if self._robot_mesh:
            self._robot_mesh.detach()
        robot_type = self._tplan.robot.robot_type
        try:
            self._robot = create_robot(
                robot_type,
                pos=self._tplan.robot.base_pos,
                rotmat=self._tplan.robot.base_rotmat,
            )
            if self._tplan.robot.start_conf is not None:
                self._robot.goto_given_conf(
                    jnt_values=self._tplan.robot.start_conf)
            self._robot_mesh = self._robot.gen_meshmodel(alpha=0.7)
            self._robot_mesh.attach_to(self.world)
            self.console.log_ok(f"Robot: {robot_type}")
        except Exception as e:
            self.console.log_warn(f"Robot spawn failed: {e}")
            traceback.print_exc()
            self._robot = None

    def _update_robot_pos(self, pos: np.ndarray):
        """Move robot base position and re-render."""
        self._tplan.robot.base_pos = pos.copy()
        if self._robot:
            self._robot.fix_to(pos=pos, rotmat=self._tplan.robot.base_rotmat)
            if self._robot_mesh:
                self._robot_mesh.detach()
            self._robot_mesh = self._robot.gen_meshmodel(alpha=0.7)
            self._robot_mesh.attach_to(self.world)

    # ==============================================================
    # FSReferencePoses — stable placement computation
    # ==============================================================
    def _compute_fs_ref_poses(self, model_path: str):
        """Compute and cache stable flat-surface reference poses."""
        if model_path in self._fs_ref_poses:
            return self._fs_ref_poses[model_path]
        if not os.path.isfile(model_path):
            self._fs_ref_poses[model_path] = []
            return []
        try:
            from wrs.manipulation.placement.flatsurface import FSReferencePoses
            obj_cmodel = mcm.CollisionModel(model_path)
            fs_rp = FSReferencePoses(obj_cmodel=obj_cmodel)
            poses = list(fs_rp)  # list of (pos, rotmat) tuples
            self._fs_ref_poses[model_path] = poses
            self.console.log_info(
                f"Computed {len(poses)} stable poses for "
                f"{Path(model_path).name}")
            return poses
        except Exception as e:
            self.console.log_warn(f"FSReferencePoses failed: {e}")
            self._fs_ref_poses[model_path] = []
            return []

    def _get_stable_rotmat(self, model_path: str) -> np.ndarray:
        """Get the most stable placement rotation for a model."""
        poses = self._compute_fs_ref_poses(model_path)
        if poses:
            return poses[0][1].copy()  # most stable (sorted by stability)
        return np.eye(3)

    def _get_stable_z_offset(self, model_path: str) -> float:
        """Get the Z offset for the most stable placement."""
        poses = self._compute_fs_ref_poses(model_path)
        if poses:
            return float(poses[0][0][2])  # Z component of pos offset
        return 0.0

    # ==============================================================
    # Staging parts & assembly ghosts
    # ==============================================================
    def _load_staging_parts(self):
        """Create staging part models from tplan staging data."""
        # Clear existing
        for cm in self._staging_models.values():
            cm.detach()
        self._staging_models.clear()
        for cm in self._ghost_models.values():
            cm.detach()
        self._ghost_models.clear()

        if not self._asmdef:
            return

        # Compute assembly world poses for ghosts
        world_poses = self._asmdef.compute_world_poses(
            fixture_pos=self._tplan.fixture_pos,
            fixture_rotmat=self._tplan.fixture_rotmat,
        )

        for pid in self._asmdef.part_ids:
            model_path = self._asmdef.model_path(pid)
            color = STAGING_COLORS[self._color_idx % len(STAGING_COLORS)]
            self._color_idx += 1
            self._staging_colors[pid] = color.copy()

            # ── Staging model (solid, draggable) ─────────────
            staging = self._tplan.get_staging(pid)
            if staging:
                s_pos = staging.pos.copy()
                s_rot = staging.rotmat.copy()
            else:
                # Default: identity rotation at a grid position on table
                idx = list(self._asmdef.part_ids).index(pid)
                s_pos = np.array([
                    -0.15 + 0.12 * (idx % 4),
                    -0.25 - 0.12 * (idx // 4),
                    self._table_height,
                ])
                s_rot = np.eye(3)
                # Store in tplan
                self._tplan.set_staging(pid, pos=s_pos, rotmat=s_rot)

            try:
                cm = mcm.CollisionModel(
                    initor=model_path, name=f"staging_{pid}")
                cm.rgba = color
                cm.pos = s_pos
                cm.rotmat = s_rot
                cm.attach_to(self.world)
                self._staging_models[pid] = cm
            except Exception as e:
                self.console.log_warn(f"Failed to load {pid}: {e}")
                # placeholder
                cm = mcm.gen_box(
                    xyz_lengths=np.array([0.04, 0.04, 0.04]),
                    pos=s_pos, rgb=color[:3], alpha=1.0)
                cm.attach_to(self.world)
                self._staging_models[pid] = cm

            # ── Assembly ghost (transparent) ─────────────────
            w_pos, w_rot = world_poses.get(pid, (np.zeros(3), np.eye(3)))
            try:
                ghost = mcm.CollisionModel(
                    initor=model_path, name=f"ghost_{pid}")
                ghost.rgba = np.array([*color[:3], GHOST_ALPHA])
                ghost.pos = w_pos
                ghost.rotmat = w_rot
                ghost.attach_to(self.world)
                self._ghost_models[pid] = ghost
            except Exception:
                pass

    def _update_assembly_ghosts(self):
        """Recompute and update assembly ghost positions from fixture."""
        if not self._asmdef:
            return
        world_poses = self._asmdef.compute_world_poses(
            fixture_pos=self._tplan.fixture_pos,
            fixture_rotmat=self._tplan.fixture_rotmat,
        )
        for pid, ghost in self._ghost_models.items():
            w_pos, w_rot = world_poses.get(pid, (np.zeros(3), np.eye(3)))
            ghost.detach()
            ghost.pos = w_pos
            ghost.rotmat = w_rot
            ghost.attach_to(self.world)

    # ==============================================================
    # Selection
    # ==============================================================
    def _select_item(self, item_type: str, item_id: str = None):
        """Select an item (robot, fixture, or part_id)."""
        # Deselect previous
        if self._selected_id and self._selected_id in self._staging_models:
            cm = self._staging_models[self._selected_id]
            cm.rgba = self._staging_colors.get(
                self._selected_id, np.array([0.7, 0.7, 0.7, 1.0]))
        self._clear_wireframe()

        self._selected_type = item_type
        self._selected_id = item_id

        # Highlight new — wireframe outline
        if item_type == "part" and item_id in self._staging_models:
            cm = self._staging_models[item_id]
            self._show_wireframe(cm)

        self._refresh_properties()
        self._refresh_parts_list()
        self._update_status()

    def _deselect_all(self):
        if self._selected_id and self._selected_id in self._staging_models:
            cm = self._staging_models[self._selected_id]
            cm.rgba = self._staging_colors.get(
                self._selected_id, np.array([0.7, 0.7, 0.7, 1.0]))
        self._clear_wireframe()
        self._selected_type = None
        self._selected_id = None
        self._refresh_properties()
        self._update_status()

    def _show_wireframe(self, cm):
        """Show a wireframe overlay on the given CollisionModel."""
        self._clear_wireframe()
        try:
            # Get the Panda3D node and create a wireframe copy
            np_node = cm.pdndp
            if np_node is None or np_node.isEmpty():
                return
            wire = np_node.copyTo(self.world.render)
            wire.setRenderModeWireframe()
            wire.setColor(0.2, 1.0, 0.4, 1.0)  # bright green wireframe
            wire.setRenderModeThickness(2)
            wire.setLightOff()
            wire.setBin("fixed", 50)  # render on top
            wire.setDepthTest(True)
            wire.setDepthWrite(False)
            self._selection_wireframe = wire
        except Exception:
            pass

    def _clear_wireframe(self):
        """Remove the wireframe selection overlay."""
        if self._selection_wireframe is not None:
            try:
                self._selection_wireframe.removeNode()
            except Exception:
                pass
            self._selection_wireframe = None

    # ==============================================================
    # Click-to-select (bounding sphere ray test)
    # ==============================================================
    def _pick_part_under_mouse(self) -> str | None:
        """Find the closest staging part under the mouse cursor.

        Uses bounding-sphere intersection for speed.
        """
        if not self.world.mouseWatcherNode.hasMouse():
            return None
        mpos = self.world.mouseWatcherNode.getMouse()
        # Build ray in world space
        near = Point3()
        far = Point3()
        self.world.camLens.extrude(mpos, near, far)
        near_w = self.world.render.getRelativePoint(self.world.cam, near)
        far_w = self.world.render.getRelativePoint(self.world.cam, far)
        ray_origin = np.array([near_w.getX(), near_w.getY(), near_w.getZ()])
        ray_dir = np.array([far_w.getX() - near_w.getX(),
                            far_w.getY() - near_w.getY(),
                            far_w.getZ() - near_w.getZ()])
        ray_len = np.linalg.norm(ray_dir)
        if ray_len < 1e-10:
            return None
        ray_dir /= ray_len

        best_pid = None
        best_t = float("inf")

        for pid, cm in self._staging_models.items():
            # Get bounding sphere center and radius from model
            center = cm.pos.copy()
            try:
                bounds = cm.pdndp.getBounds()
                radius = bounds.getRadius()
                bc = bounds.getCenter()
                center = np.array([bc.getX(), bc.getY(), bc.getZ()])
                # Convert from local to world
                world_center = cm.pdndp.getPos(self.world.render)
                center = np.array([world_center.getX(),
                                   world_center.getY(),
                                   world_center.getZ()])
            except Exception:
                radius = 0.05  # fallback

            # Ray-sphere intersection: |O + tD - C|^2 = r^2
            oc = ray_origin - center
            a = np.dot(ray_dir, ray_dir)  # should be 1
            b = 2.0 * np.dot(oc, ray_dir)
            c = np.dot(oc, oc) - radius * radius
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            t = (-b - np.sqrt(disc)) / (2 * a)
            if t < 0:
                t = (-b + np.sqrt(disc)) / (2 * a)
            if t < 0:
                continue
            if t < best_t:
                best_t = t
                best_pid = pid

        return best_pid

    # ==============================================================
    # Mouse-to-world (constrained to table plane)
    # ==============================================================
    def _mouse_to_table(self) -> np.ndarray | None:
        """Ray-cast mouse to table plane (Z = table_height)."""
        if not self.world.mouseWatcherNode.hasMouse():
            return None
        mpos = self.world.mouseWatcherNode.getMouse()
        near = Point3()
        far = Point3()
        self.world.camLens.extrude(mpos, near, far)
        near_world = self.world.render.getRelativePoint(
            self.world.cam, near)
        far_world = self.world.render.getRelativePoint(
            self.world.cam, far)
        direction = far_world - near_world
        dz = direction.getZ()
        if abs(dz) < 1e-8:
            return None
        t = (self._table_height - near_world.getZ()) / dz
        if t < 0:
            return None
        hit = near_world + direction * t
        return np.array([hit.getX(), hit.getY(), self._table_height])

    # ==============================================================
    # Per-frame update
    # ==============================================================
    def _editor_update(self, task):
        mode = self.transform.mode
        if mode == TransformMode.NONE:
            return task.cont

        if mode == TransformMode.GRAB:
            self._update_grab()
        elif mode == TransformMode.ROTATE:
            self._update_rotate()

        return task.cont

    def _update_grab(self):
        """Move selected item along table surface."""
        if not self._selected_type:
            return
        hit = self._mouse_to_table()
        if hit is None:
            return
        pos, rotmat = self.transform.update_grab(hit)
        self._apply_transform(pos, rotmat)

    def _apply_transform(self, pos, rotmat):
        """Apply a transform result to the selected item."""
        if self._selected_type == "part" and self._selected_id:
            cm = self._staging_models.get(self._selected_id)
            if cm:
                cm.detach()
                cm.pos = pos
                cm.rotmat = rotmat
                cm.attach_to(self.world)
                self._refresh_properties()
        elif self._selected_type == "robot":
            self._update_robot_pos(pos)
            self._refresh_properties()
        elif self._selected_type == "fixture":
            self._tplan.fixture_pos = pos.copy()
            self._draw_fixture_marker()
            self._update_assembly_ghosts()
            self._refresh_properties()

    def _update_rotate(self):
        """Rotate selected item."""
        if not self._selected_type or self._selected_type != "part":
            return
        if not self.world.mouseWatcherNode.hasMouse():
            return
        mpos = self.world.mouseWatcherNode.getMouse()
        pos, rotmat = self.transform.update_rotate(
            (mpos.getX(), mpos.getY()))
        self._apply_transform(pos, rotmat)

    # ==============================================================
    # Key bindings
    # ==============================================================
    def _setup_keys(self):
        w = self.world
        w.accept("g", self._key_grab)
        w.accept("r", self._key_rotate)
        w.accept("x", lambda: self._key_axis(AxisConstraint.X))
        w.accept("y", lambda: self._key_axis(AxisConstraint.Y))
        w.accept("z", lambda: self._key_axis(AxisConstraint.Z))
        w.accept("escape", self._key_escape)
        w.accept("mouse1", self._key_confirm_or_pick)
        w.accept("control-s", self._on_save_tplan)
        w.accept("control-o", self._on_load_file)
        # Cycle stable placements
        w.accept("tab", self._key_cycle_placement)

    def _key_grab(self):
        if not self._selected_type:
            return
        pos = self._get_selected_pos()
        rotmat = self._get_selected_rotmat()
        if pos is None:
            return
        # Use the current selected position as the initial world hit
        hit = self._mouse_to_table()
        if hit is None:
            hit = pos.copy()
        self.transform.start_grab(pos, rotmat, hit)
        self._update_status()

    def _key_rotate(self):
        if self._selected_type != "part":
            return
        pos = self._get_selected_pos()
        rotmat = self._get_selected_rotmat()
        if pos is None:
            return
        if not self.world.mouseWatcherNode.hasMouse():
            return
        mpos = self.world.mouseWatcherNode.getMouse()
        self.transform.start_rotate(
            pos, rotmat, (mpos.getX(), mpos.getY()))
        self._update_status()

    def _key_axis(self, axis):
        self.transform.constrain(axis)
        self._update_status()

    def _key_escape(self):
        if self.transform.mode != TransformMode.NONE:
            pos, rotmat = self.transform.cancel()
            if pos is not None:
                self._apply_transform(pos, rotmat)
            self._refresh_properties()
        else:
            self._deselect_all()
        self._update_status()

    def _key_confirm_or_pick(self):
        if self.transform.mode != TransformMode.NONE:
            pos, rotmat = self.transform.confirm()
            if pos is not None:
                self._apply_transform(pos, rotmat)
            self._sync_to_tplan()
            # Re-show wireframe at new position
            if self._selected_type == "part" and self._selected_id:
                cm = self._staging_models.get(self._selected_id)
                if cm:
                    self._show_wireframe(cm)
            self._update_status()
            self.console.log_ok("Transform applied.")
        else:
            # Click-to-select: pick part under mouse
            pid = self._pick_part_under_mouse()
            if pid:
                self._select_item("part", pid)
            else:
                self._deselect_all()

    def _key_cycle_placement(self):
        """Cycle through stable placements for the selected part."""
        if self._selected_type != "part" or not self._selected_id:
            return
        if not self._asmdef:
            return
        model_path = self._asmdef.model_path(self._selected_id)
        poses = self._compute_fs_ref_poses(model_path)
        if not poses:
            self.console.log_warn("No stable placements computed.")
            return

        cm = self._staging_models.get(self._selected_id)
        if not cm:
            return

        # Find current pose index
        current_rot = cm.rotmat
        best_idx = 0
        best_dist = float("inf")
        for i, (p, r) in enumerate(poses):
            dist = np.linalg.norm(r - current_rot)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        # Cycle to next
        next_idx = (best_idx + 1) % len(poses)
        new_rot = poses[next_idx][1].copy()
        z_off = float(poses[next_idx][0][2])
        new_pos = cm.pos.copy()
        new_pos[2] = self._table_height + z_off

        cm.detach()
        cm.pos = new_pos
        cm.rotmat = new_rot
        cm.attach_to(self.world)
        self._show_wireframe(cm)
        self._current_placement_idx[self._selected_id] = next_idx
        self._sync_to_tplan()
        self._refresh_properties()
        self.console.log_info(
            f"Placement {next_idx + 1}/{len(poses)} for {self._selected_id}")

    # ==============================================================
    # Helpers
    # ==============================================================
    def _get_selected_pos(self) -> np.ndarray | None:
        if self._selected_type == "part" and self._selected_id:
            cm = self._staging_models.get(self._selected_id)
            return cm.pos.copy() if cm else None
        elif self._selected_type == "robot":
            return self._tplan.robot.base_pos.copy()
        elif self._selected_type == "fixture":
            return self._tplan.fixture_pos.copy()
        return None

    def _get_selected_rotmat(self) -> np.ndarray | None:
        if self._selected_type == "part" and self._selected_id:
            cm = self._staging_models.get(self._selected_id)
            return cm.rotmat.copy() if cm else None
        elif self._selected_type == "robot":
            return self._tplan.robot.base_rotmat.copy()
        elif self._selected_type == "fixture":
            return self._tplan.fixture_rotmat.copy()
        return None

    def _sync_to_tplan(self):
        """Push current 3D state back into the TaskPlan."""
        for pid, cm in self._staging_models.items():
            self._tplan.set_staging(pid, pos=cm.pos, rotmat=cm.rotmat)

    def _on_mode_change(self, mode, axis):
        self._update_status()

    # ==============================================================
    # Collision checking API (stub)
    # ==============================================================
    def check_staging_collisions(self) -> list[tuple[str, str]]:
        """Check for collisions between staging parts.

        Returns list of (part_id_a, part_id_b) collision pairs.
        (Not yet implemented — stub for future use.)
        """
        # TODO: implement collision checking between staging models
        return []

    def check_robot_reachability(self, part_id: str) -> bool:
        """Check if the robot can reach a staging position.

        (Not yet implemented — stub for future use.)
        """
        # TODO: implement robot IK reachability check
        return True

    # ==============================================================
    # Grasp visualization (selective)
    # ==============================================================
    def _show_grasp_for_step(self, step_id: int):
        """Show the selected grasp for a step (stub).

        TODO: Load grasp collection, show gripper at grasp pose.
        """
        params = self._tplan.get_step_params(step_id)
        if params and params.grasp_id is not None:
            self.console.log_info(
                f"Grasp viz: step {step_id}, grasp_id={params.grasp_id} "
                f"(not yet rendered)")
        else:
            self.console.log_info(
                f"Step {step_id}: no grasp_id set (auto-select)")

    # ==============================================================
    # GUI: Right panel
    # ==============================================================
    def _build_right_panel(self):
        pw = RIGHT_W
        panel_h = 2.0
        inner_w = pw * 2 - LEFT_MARGIN * 2
        self._right_frame = create_panel(
            self.world.a2dTopRight, pw * 2, panel_h,
            (-pw * 2, 0), bg=BG_MID,
        )
        y = -TOP_MARGIN * 2
        x0 = LEFT_MARGIN

        # Title
        create_section_header(self._right_frame, "Task Plan", (x0, y))
        y -= SECTION_SIZE * 1.8

        # Parts list
        create_label(self._right_frame, "Parts (staging):", (x0, y),
                     color=TEXT_SECONDARY)
        y -= TEXT_SIZE * 1.3
        list_h = 0.35
        self._parts_scroll = DirectScrolledFrame(
            canvasSize=(0, inner_w, -0.01, list_h),
            frameSize=(0, inner_w, -list_h, 0),
            frameColor=LIST_BG,
            pos=LPoint3f(LEFT_MARGIN, 0, y),
            parent=self._right_frame,
            scrollBarWidth=0.015,
            autoHideScrollBars=True,
        )
        y -= list_h + TOP_MARGIN * 2

        # Steps list — with separator and extra spacing
        create_separator(self._right_frame, (LEFT_MARGIN, y), inner_w)
        y -= TOP_MARGIN * 2
        create_label(self._right_frame, "Step Params:", (x0, y),
                     color=TEXT_SECONDARY)
        y -= TEXT_SIZE * 1.3
        list_h2 = 0.22
        self._steps_scroll = DirectScrolledFrame(
            canvasSize=(0, inner_w, -0.01, list_h2),
            frameSize=(0, inner_w, -list_h2, 0),
            frameColor=LIST_BG,
            pos=LPoint3f(LEFT_MARGIN, 0, y),
            parent=self._right_frame,
            scrollBarWidth=0.015,
            autoHideScrollBars=True,
        )
        y -= list_h2 + TOP_MARGIN

        # Buttons — centered in panel
        create_separator(self._right_frame, (LEFT_MARGIN, y), inner_w)
        y -= TOP_MARGIN * 2
        # Center two buttons in the panel
        btn_w = inner_w * 0.45
        btn_gap = inner_w * 0.06
        btn_x0 = LEFT_MARGIN + btn_w * 0.5 + 0.01
        btn_x1 = btn_x0 + btn_w + btn_gap

        # Brighter button color for secondary buttons
        sec_bg = (0.30, 0.32, 0.38, 1.0)
        sec_hover = (0.40, 0.42, 0.50, 1.0)

        create_accent_button(self._right_frame, "Load",
                             (btn_x0, y), self._on_load_file,
                             width=btn_w)
        create_accent_button(self._right_frame, "Save",
                             (btn_x1, y), self._on_save_tplan,
                             width=btn_w)
        y -= TEXT_SIZE * 2.4

        create_button(self._right_frame, "New",
                      (btn_x0, y), self._on_new, width=btn_w,
                      bg=sec_bg, hover_bg=sec_hover)
        create_button(self._right_frame, "Sel Robot",
                      (btn_x1, y),
                      lambda: self._select_item("robot"),
                      width=btn_w,
                      bg=sec_bg, hover_bg=sec_hover)
        y -= TEXT_SIZE * 2.4

        create_button(self._right_frame, "Sel Fixture",
                      (btn_x0, y),
                      lambda: self._select_item("fixture"),
                      width=btn_w,
                      bg=sec_bg, hover_bg=sec_hover)

    # ==============================================================
    # GUI: Left panel (properties)
    # ==============================================================
    def _build_left_panel(self):
        pw = LEFT_W
        panel_h = 2.0
        inner_w = pw * 2 - LEFT_MARGIN * 2
        self._left_frame = create_panel(
            self.world.a2dTopLeft, pw * 2, panel_h,
            (0, 0), bg=BG_MID,
        )
        y = -TOP_MARGIN * 2
        x0 = LEFT_MARGIN
        label_x = x0 + 0.01
        val_x = 0.13        # value labels (wider to avoid overlap)
        entry_x = 0.13      # entry fields
        entry_w = 0.12      # entry width

        # Section header
        create_section_header(self._left_frame, "Properties", (x0, y))
        y -= SECTION_SIZE * 1.8

        # ── Selected item info ───────────────────────────────
        create_label(self._left_frame, "Selected:", (label_x, y),
                     color=TEXT_SECONDARY, scale=SMALL_TEXT)
        y -= TEXT_SIZE * 1.4
        self._prop_type = create_label(
            self._left_frame, "(none)", (label_x + 0.01, y),
            color=TEXT_PRIMARY, scale=SMALL_TEXT)
        y -= TEXT_SIZE * 1.6

        create_label(self._left_frame, "ID:", (label_x, y),
                     color=TEXT_SECONDARY, scale=SMALL_TEXT)
        self._prop_id = create_label(
            self._left_frame, "-", (val_x, y),
            color=TEXT_ACCENT, scale=SMALL_TEXT)
        y -= TEXT_SIZE * 2.0

        create_separator(self._left_frame, (LEFT_MARGIN, y), inner_w)
        y -= TOP_MARGIN * 2

        # ── Position ─────────────────────────────────────────
        create_label(self._left_frame, "Position", (label_x, y),
                     color=TEXT_SECONDARY, scale=SMALL_TEXT)
        y -= TEXT_SIZE * 1.5

        self._pos_entries = {}
        for axis_name, col in [("X", (0.95, 0.30, 0.30, 1)),
                               ("Y", (0.30, 0.85, 0.30, 1)),
                               ("Z", (0.30, 0.55, 0.95, 1))]:
            create_label(self._left_frame, f"{axis_name}:",
                         (label_x, y), color=col, scale=SMALL_TEXT)
            e = create_entry(self._left_frame, (entry_x, y),
                             width=entry_w, initial="0.000")
            self._pos_entries[axis_name] = e
            y -= TEXT_SIZE * 1.6
        y -= TEXT_SIZE * 0.5

        # ── Table height ─────────────────────────────────────
        create_separator(self._left_frame, (LEFT_MARGIN, y), inner_w)
        y -= TOP_MARGIN * 2

        create_label(self._left_frame, "Table H:", (label_x, y),
                     color=TEXT_SECONDARY, scale=SMALL_TEXT)
        self._table_h_entry = create_entry(
            self._left_frame, (entry_x, y),
            width=entry_w, initial="0.000")
        y -= TEXT_SIZE * 2.2

        # ── Primitive type ───────────────────────────────────
        create_label(self._left_frame, "Primitive:", (label_x, y),
                     color=TEXT_SECONDARY, scale=SMALL_TEXT)
        y -= TEXT_SIZE * 1.4
        self._prim_label = create_label(
            self._left_frame, "-", (label_x + 0.01, y),
            color=TEXT_PRIMARY, scale=SMALL_TEXT)
        y -= TEXT_SIZE * 2.0

        # ── Placement dropdown (shown when part selected) ────
        create_separator(self._left_frame, (LEFT_MARGIN, y), inner_w)
        y -= TOP_MARGIN * 2

        self._placement_label = create_label(
            self._left_frame, "Placement:", (label_x, y),
            color=TEXT_SECONDARY, scale=SMALL_TEXT)
        y -= TEXT_SIZE * 1.5
        self._placement_menu = DirectOptionMenu(
            text="(none)",
            scale=SMALL_TEXT,
            items=["(none)"],
            initialitem=0,
            highlightColor=(0.28, 0.56, 0.92, 0.65),
            frameColor=(0.22, 0.23, 0.27, 1.0),
            text_fg=TEXT_PRIMARY,
            pos=LPoint3f(label_x, 0, y),
            parent=self._left_frame,
            command=self._on_placement_changed,
            textMayChange=1,
        )
        # Initially hidden
        self._placement_label.hide()
        self._placement_menu.hide()
        y -= TEXT_SIZE * 2.4

        # ── Apply button ─────────────────────────────────────
        apply_w = inner_w * 0.5
        create_accent_button(
            self._left_frame, "Apply",
            (LEFT_MARGIN + apply_w * 0.5 + 0.01, y),
            self._on_apply_properties,
            width=apply_w)

    # ==============================================================
    # GUI: Status bar + Console
    # ==============================================================
    def _build_status_bar(self):
        bar_y = -1.0
        self._status_frame = DirectFrame(
            parent=self.world.aspect2d,
            frameColor=(*BG_DARK[:3], 0.95),
            frameSize=(-2, 2, bar_y, bar_y + STATUS_H),
            pos=(0, 0, 0),
        )
        self._status_label = DirectLabel(
            parent=self._status_frame,
            text="Mode: Select  |  Item: none",
            text_fg=TEXT_SECONDARY,
            text_scale=SMALL_TEXT,
            text_align=TextNode.ALeft,
            pos=(-1 + LEFT_W * 2 + 0.02, 0,
                 bar_y + STATUS_H * 0.25),
            frameColor=(0, 0, 0, 0),
        )

    def _build_console(self):
        console_w = 1.0 - LEFT_W - RIGHT_W
        self.console = ConsoleWindow(
            parent=self.world.a2dBottomCenter,
            width=console_w,
            height=CONSOLE_H,
            y_offset=STATUS_H,
        )

    def _update_status(self):
        mode = self.transform.mode.name.capitalize()
        axis = self.transform.axis
        ax_str = f" [{axis.name}]" if axis != AxisConstraint.FREE else ""
        item_str = "none"
        if self._selected_type == "part":
            item_str = f"Part: {self._selected_id}"
        elif self._selected_type == "robot":
            item_str = "Robot"
        elif self._selected_type == "fixture":
            item_str = "Fixture"
        self._status_label["text"] = (
            f"Mode: {mode}{ax_str}  |  {item_str}")

    # ==============================================================
    # Properties panel refresh
    # ==============================================================
    def _refresh_properties(self):
        if not self._selected_type:
            self._prop_type["text"] = "(none)"
            self._prop_id["text"] = "-"
            for e in self._pos_entries.values():
                e.set("0.000")
            self._prim_label["text"] = "-"
            # Hide placement dropdown
            if self._placement_label:
                self._placement_label.hide()
            if self._placement_menu:
                self._placement_menu.hide()
            return

        self._prop_type["text"] = self._selected_type.capitalize()
        self._prop_id["text"] = self._selected_id or "-"

        pos = self._get_selected_pos()
        if pos is not None:
            for i, axis in enumerate(["X", "Y", "Z"]):
                self._pos_entries[axis].set(f"{pos[i]:.4f}")

        # Primitive for selected part
        if self._selected_type == "part" and self._asmdef:
            # Find step for this part
            for step in self._asmdef.steps:
                if step.part_id == self._selected_id:
                    params = self._tplan.get_step_params(step.step_id)
                    if params:
                        self._prim_label["text"] = params.primitive.value
                    else:
                        self._prim_label["text"] = "auto"
                    break
            # Show & populate placement dropdown
            self._populate_placement_menu()
        else:
            self._prim_label["text"] = "-"
            # Hide placement dropdown
            if self._placement_label:
                self._placement_label.hide()
            if self._placement_menu:
                self._placement_menu.hide()

    def _populate_placement_menu(self):
        """Populate the placement dropdown for the selected part.

        Only uses cached poses. Does NOT trigger computation
        (FSReferencePoses uses ODE which can segfault during init).
        """
        if not self._placement_menu or not self._placement_label:
            return
        if self._selected_type != "part" or not self._selected_id:
            self._placement_label.hide()
            self._placement_menu.hide()
            return

        model_path = self._asmdef.model_path(self._selected_id)
        # Only use cached poses, don't compute
        poses = self._fs_ref_poses.get(model_path, None)

        self._placement_label.show()
        self._placement_menu.show()

        if poses is None:
            # Not yet computed — show option to compute
            self._placement_menu["items"] = ["(press Tab to compute)"]
            self._placement_menu.set(0)
            return

        if not poses:
            self._placement_menu["items"] = ["(no stable poses)"]
            self._placement_menu.set(0)
            return

        # Build items list
        items = []
        for i, (p, r) in enumerate(poses):
            items.append(f"Pose {i} (z={p[2]:.3f})")

        self._placement_menu["items"] = items

        # Try to find current matching pose
        pid = self._selected_id
        cur_idx = self._current_placement_idx.get(pid, 0)
        if cur_idx >= len(items):
            cur_idx = 0
        self._placement_menu.set(cur_idx)

    def _on_placement_changed(self, choice):
        """Callback when placement dropdown selection changes."""
        if self._selected_type != "part" or not self._selected_id:
            return
        if not self._asmdef:
            return

        # Parse index from choice string
        try:
            idx = int(choice.split(" ")[1])
        except (IndexError, ValueError):
            return

        model_path = self._asmdef.model_path(self._selected_id)
        poses = self._compute_fs_ref_poses(model_path)
        if not poses or idx >= len(poses):
            return

        cm = self._staging_models.get(self._selected_id)
        if not cm:
            return

        new_rot = poses[idx][1].copy()
        z_off = float(poses[idx][0][2])
        new_pos = cm.pos.copy()
        new_pos[2] = self._table_height + z_off

        cm.detach()
        cm.pos = new_pos
        cm.rotmat = new_rot
        cm.attach_to(self.world)

        # Update wireframe
        self._show_wireframe(cm)
        self._current_placement_idx[self._selected_id] = idx
        self._sync_to_tplan()

        # Update position entries
        for i, axis in enumerate(["X", "Y", "Z"]):
            self._pos_entries[axis].set(f"{new_pos[i]:.4f}")

        self.console.log_info(
            f"Placement {idx}/{len(poses)} for {self._selected_id}")

    def _on_apply_properties(self):
        """Apply manual position entries."""
        try:
            pos = np.array([
                float(self._pos_entries["X"].get()),
                float(self._pos_entries["Y"].get()),
                float(self._pos_entries["Z"].get()),
            ])
        except ValueError:
            self.console.log_warn("Invalid position values.")
            return

        # Apply table height
        try:
            new_h = float(self._table_h_entry.get())
            if new_h != self._table_height:
                self._table_height = new_h
                self._draw_table()
                self.console.log_info(f"Table height: {new_h:.4f}")
        except ValueError:
            pass

        if self._selected_type == "part" and self._selected_id:
            cm = self._staging_models.get(self._selected_id)
            if cm:
                cm.detach()
                cm.pos = pos
                cm.attach_to(self.world)
                self._show_wireframe(cm)
                self._sync_to_tplan()
                self.console.log_ok(
                    f"Applied pos to {self._selected_id}")
        elif self._selected_type == "robot":
            self._update_robot_pos(pos)
            self.console.log_ok("Applied robot position.")
        elif self._selected_type == "fixture":
            self._tplan.fixture_pos = pos.copy()
            self._draw_fixture_marker()
            self._update_assembly_ghosts()
            self.console.log_ok("Applied fixture position.")

    # ==============================================================
    # Parts list and steps list
    # ==============================================================
    def _refresh_parts_list(self):
        canvas = self._parts_scroll.getCanvas()
        for child in canvas.getChildren():
            child.removeNode()
        if not self._asmdef:
            return
        pw = RIGHT_W
        inner_w = pw * 2 - LEFT_MARGIN * 2
        y = -SMALL_TEXT * 1.0
        x0 = LEFT_MARGIN * 0.5
        for pid in self._asmdef.part_ids:
            is_sel = (self._selected_type == "part"
                      and self._selected_id == pid)
            pdef = self._asmdef.get_part(pid)
            label = f"{('> ' if is_sel else '  ')}{pdef.name}"
            create_list_item(
                canvas, label, (x0, y), width=inner_w,
                on_click=self._on_pick_part,
                item_id=pid, selected=is_sel,
            )
            y -= SMALL_TEXT * 1.8
        self._parts_scroll["canvasSize"] = (0, inner_w, y, 0)

    def _refresh_steps_list(self):
        canvas = self._steps_scroll.getCanvas()
        for child in canvas.getChildren():
            child.removeNode()
        if not self._asmdef:
            return
        pw = RIGHT_W
        inner_w = pw * 2 - LEFT_MARGIN * 2
        y = -SMALL_TEXT * 1.0
        x0 = LEFT_MARGIN * 0.5
        for step in self._asmdef.steps:
            params = self._tplan.get_step_params(step.step_id)
            prim_str = params.primitive.value if params else "auto"
            label = f"S{step.step_id}: {step.part_id} [{prim_str}]"
            create_list_item(
                canvas, label, (x0, y), width=inner_w,
                on_click=self._on_pick_step,
                item_id=step.step_id,
            )
            y -= SMALL_TEXT * 1.8
        self._steps_scroll["canvasSize"] = (0, inner_w, y, 0)

    def _on_pick_part(self, part_id):
        """Callback from parts list item click."""
        self._select_item("part", part_id)

    def _on_pick_step(self, step_id):
        """Callback from steps list item click."""
        self._on_select_step(step_id)

    def _on_select_step(self, step_id: int):
        """Select a step and its associated part."""
        if not self._asmdef:
            return
        for step in self._asmdef.steps:
            if step.step_id == step_id:
                self._select_item("part", step.part_id)
                self._show_grasp_for_step(step_id)
                break

    # ==============================================================
    # File I/O
    # ==============================================================
    def _on_load_file(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            fp = filedialog.askopenfilename(
                title="Load Assembly / Task Plan",
                filetypes=[
                    ("All supported", "*.asmdef *.tplan"),
                    ("Assembly Def", "*.asmdef"),
                    ("Task Plan", "*.tplan"),
                    ("All", "*.*"),
                ],
            )
            root.destroy()
            if fp:
                ext = os.path.splitext(fp)[1].lower()
                if ext == ".tplan":
                    self._do_load_tplan(fp)
                elif ext == ".asmdef":
                    self._do_load_asmdef(fp)
                else:
                    self.console.log_warn(f"Unknown file: {ext}")
        except Exception as e:
            self.console.log_warn(f"Load error: {e}")

    def _do_load_asmdef(self, filepath: str):
        """Load an .asmdef and create a fresh task plan."""
        try:
            asmdef = AssemblyDef.load(filepath)
            asmdef.validate(strict=True)
        except Exception as e:
            self.console.log_warn(f"Failed to load asmdef: {e}")
            traceback.print_exc()
            return

        self._asmdef = asmdef
        self._tplan = TaskPlan(
            assembly_file=os.path.abspath(filepath),
            name=f"{asmdef.name} Plan",
        )
        # Default step params
        for step in asmdef.steps:
            self._tplan.set_step_params(StepParams(step_id=step.step_id))

        self._color_idx = 0
        print("[TplanEditor] Spawning robot...")
        self._spawn_robot()
        print("[TplanEditor] Drawing fixture...")
        self._draw_fixture_marker()
        print("[TplanEditor] Loading staging parts...")
        self._load_staging_parts()
        print("[TplanEditor] Refreshing GUI...")
        self._refresh_parts_list()
        self._refresh_steps_list()
        self._refresh_properties()
        self._update_status()
        print("[TplanEditor] Done!")
        self.console.log_ok(
            f"Loaded .asmdef: {asmdef.name} ({asmdef.n_parts} parts)")

    def _do_load_tplan(self, filepath: str):
        """Load an existing .tplan."""
        try:
            tplan = TaskPlan.load(filepath)
        except Exception as e:
            self.console.log_warn(f"Failed to load tplan: {e}")
            traceback.print_exc()
            return

        self._tplan = tplan
        self._current_file = filepath

        # Load linked assembly
        asmdef = tplan.assembly
        if asmdef is None:
            self.console.log_warn(
                f"Cannot load assembly: {tplan.assembly_file}")
            return
        self._asmdef = asmdef

        self._color_idx = 0
        self._spawn_robot()
        self._draw_fixture_marker()
        self._load_staging_parts()
        self._refresh_parts_list()
        self._refresh_steps_list()
        self._refresh_properties()
        self._update_status()
        self.console.log_ok(
            f"Loaded .tplan: {tplan.name} "
            f"(asm={asmdef.name}, {len(tplan.staging)} staging)")

    def _on_save_tplan(self):
        self._sync_to_tplan()
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            fp = filedialog.asksaveasfilename(
                title="Save Task Plan",
                defaultextension=".tplan",
                filetypes=[
                    ("Task Plan", "*.tplan"),
                    ("All", "*.*"),
                ],
                initialfile=self._current_file or "plan.tplan",
            )
            root.destroy()
            if fp:
                self._tplan.save(fp)
                self._current_file = fp
                self.console.log_ok(f"Saved: {fp}")
        except Exception as e:
            self.console.log_warn(f"Save error: {e}")

    def _on_new(self):
        # Clear scene
        for cm in self._staging_models.values():
            cm.detach()
        self._staging_models.clear()
        for cm in self._ghost_models.values():
            cm.detach()
        self._ghost_models.clear()
        if self._robot_mesh:
            self._robot_mesh.detach()
        if self._fixture_marker:
            self._fixture_marker.detach()

        self._asmdef = None
        self._tplan = TaskPlan(name="New Task Plan")
        self._current_file = None
        self._deselect_all()
        self._refresh_parts_list()
        self._refresh_steps_list()
        self._refresh_properties()
        self._update_status()
        self.console.log_info("New task plan created.")

    # ==============================================================
    # Run
    # ==============================================================
    def run(self):
        self.world.run()
