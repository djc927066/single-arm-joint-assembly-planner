"""
CollisionBox Parameter Editor
==============================
Standalone tool for visually adjusting CollisionBox parameters on a 3D model.

Usage:
    python collision_box_editor.py
    python -m sealp.editor.collision_box_editor

Load an STL model file, add/remove CollisionBox primitives, tweak their
center and half-extent parameters in real time, then generate a ready-to-paste
``_cdprimitive_fn`` function.

Created on 2026/03/23
Author: Hao Chen (chen960216@gmail.com)
"""
import os
import sys
import traceback
import numpy as np
from panda3d.core import (CollisionNode, CollisionBox, Point3,
                          TextNode, TransparencyAttrib)
from direct.gui.DirectGui import (DirectFrame, DirectLabel, DirectButton,
                                  DirectEntry, DGG, DirectScrolledList)
# ---------------------------------------------------------------------------
# Bootstrap: make sure the project root is on sys.path so imports work
# when this file is executed directly.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import wrs.modeling.geometric_model as mgm
import wrs.basis.robot_math as rm
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEXT_SCALE = 0.04
BTN_SCALE = (0.065, 1, 0.065)
ENTRY_W = 8
PANEL_CLR = (0.12, 0.12, 0.16, 0.92)
LABEL_CLR = (0, 0, 0, 0)
BOX_COLORS = [
    [1.0, 0.3, 0.3, 0.35],
    [0.3, 1.0, 0.3, 0.35],
    [0.3, 0.3, 1.0, 0.35],
    [1.0, 1.0, 0.3, 0.35],
    [1.0, 0.3, 1.0, 0.35],
    [0.3, 1.0, 1.0, 0.35],
    [1.0, 0.6, 0.2, 0.35],
    [0.6, 0.2, 1.0, 0.35],
]


def _make_btn(text, command, parent, pos=(0, 0, 0), scale=BTN_SCALE,
              frame_size=(-1, 1, -0.3, 0.6)):
    """Create a small flat button."""
    return DirectButton(
        text=text,
        text_scale=0.4,
        text_fg=(1, 1, 1, 1),
        text_pos=(0, 0),
        frameColor=((0.28, 0.47, 0.78, 1),
                    (0.85, 0.35, 0.35, 1),
                    (0.18, 0.32, 0.58, 1),
                    (0.5, 0.5, 0.5, 1)),
        relief=DGG.FLAT,
        frameSize=frame_size,
        command=command,
        parent=parent,
        pos=pos,
        scale=scale,
    )


class CollisionBoxEditor:
    """Main editor class — creates a Panda3D window with editing GUI."""

    def __init__(self):
        import wrs.visualization.panda.world as wd
        self.base = wd.World(cam_pos=[1.5, -1.5, 1.0],
                             lookat_pos=[0.3, 0, 0.1])
        # Frame displayed at origin for reference
        mgm.gen_frame(ax_length=0.15).attach_to(self.base)
        # Internal state
        self._model_gm = None  # GeometricModel for the loaded mesh
        self._box_data = []  # list of dicts {cx,cy,cz, hx,hy,hz}
        self._box_visuals = []  # list of GeometricModel boxes
        self._box_cd_np = None  # Panda3D NodePath for collision preview
        self._selected_idx = -1  # currently selected box index
        self._build_ui()
        self.base.run()

    # ==================================================================
    # UI construction
    # ==================================================================
    def _build_ui(self):
        a2d = self.base.aspect2d
        # --- Top bar: model path entry + Load button + status ----------
        self._top_frame = DirectFrame(
            frameSize=(-1.78, 1.78, -0.08, 0.08),
            frameColor=PANEL_CLR,
            pos=(0, 0, 0.92),
            parent=a2d,
        )
        DirectLabel(
            text="Model Path:",
            text_scale=TEXT_SCALE,
            text_fg=(0.9, 0.9, 0.9, 1),
            text_align=TextNode.ALeft,
            frameColor=LABEL_CLR,
            pos=(-1.72, 0, -0.015),
            parent=self._top_frame,
        )
        self._path_entry = DirectEntry(
            text="",
            initialText="",
            scale=TEXT_SCALE / 1.25,
            width=30,
            numLines=1,
            frameColor=(1, 1, 1, 1),
            borderWidth=(0, 0),
            overflow=True,
            text_fg=(0, 0, 0, 1),
            pos=(-1.20, 0, -0.015),
            command=self._on_load_model,
            focusOutCommand=lambda: None,
            parent=self._top_frame,
        )
        _make_btn("Load", self._on_load_btn, self._top_frame,
                  pos=(0.58, 0, -0.015))
        self._status_label = DirectLabel(
            text="Ready — enter model path & click Load",
            text_scale=TEXT_SCALE * 0.85,
            text_fg=(0.7, 0.9, 0.7, 1),
            text_align=TextNode.ALeft,
            frameColor=LABEL_CLR,
            pos=(0.74, 0, -0.015),
            parent=self._top_frame,
        )
        # --- Left panel: box list + Add/Remove -------------------------
        self._left_frame = DirectFrame(
            frameSize=(-0.32, 0.32, -0.90, 0.0),
            frameColor=PANEL_CLR,
            pos=(-1.46, 0, 0.80),
            parent=a2d,
        )
        DirectLabel(
            text="CollisionBoxes",
            text_scale=TEXT_SCALE,
            text_fg=(1, 1, 1, 1),
            frameColor=LABEL_CLR,
            pos=(0, 0, -0.05),
            parent=self._left_frame,
        )
        _make_btn("+ Add", self._add_box, self._left_frame,
                  pos=(-0.14, 0, -0.12))
        _make_btn("- Remove", self._remove_box, self._left_frame,
                  pos=(0.14, 0, -0.12))
        self._box_buttons = []
        self._box_list_frame = DirectFrame(
            frameSize=(-0.30, 0.30, -0.70, 0.0),
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, -0.18),
            parent=self._left_frame,
        )
        # --- Right panel: parameter editing ----------------------------
        self._right_frame = DirectFrame(
            frameSize=(-0.42, 0.42, -0.90, 0.0),
            frameColor=PANEL_CLR,
            pos=(1.36, 0, 0.80),
            parent=a2d,
        )
        self._param_title = DirectLabel(
            text="No box selected",
            text_scale=TEXT_SCALE,
            text_fg=(1, 1, 1, 1),
            frameColor=LABEL_CLR,
            pos=(0, 0, -0.05),
            parent=self._right_frame,
        )
        self._param_entries = {}
        labels = [
            ("center_x", "Center X:"),
            ("center_y", "Center Y:"),
            ("center_z", "Center Z:"),
            ("half_x", "Half X:"),
            ("half_y", "Half Y:"),
            ("half_z", "Half Z:"),
        ]
        acc = -0.12
        for key, lbl_text in labels:
            DirectLabel(
                text=lbl_text,
                text_scale=TEXT_SCALE * 0.9,
                text_fg=(0.85, 0.85, 0.85, 1),
                text_align=TextNode.ALeft,
                frameColor=LABEL_CLR,
                pos=(-0.38, 0, acc),
                parent=self._right_frame,
            )
            entry = DirectEntry(
                text="",
                scale=TEXT_SCALE * 0.9 / 1.25,
                width=ENTRY_W,
                command=lambda txt, k=key: self._on_param_changed(k, txt),
                focusOutCommand=lambda k=key: self._on_param_focus_out(k),
                initialText="",
                numLines=1,
                frameColor=(1, 1, 1, 1),
                borderWidth=(0, 0),
                overflow=True,
                text_fg=(0, 0, 0, 1),
                pos=(-0.02, 0, acc),
                parent=self._right_frame,
            )
            entry.enterText("0.0")
            self._param_entries[key] = entry
            acc -= TEXT_SCALE * 2.2
        # Apply button
        acc -= TEXT_SCALE
        _make_btn("Apply", self._apply_params, self._right_frame,
                  pos=(-0.12, 0, acc))
        _make_btn("Gen Code", self._generate_code, self._right_frame,
                  pos=(0.16, 0, acc))
        # radius entry (for expand_radius parameter)
        acc -= TEXT_SCALE * 3
        DirectLabel(
            text="radius (expand):",
            text_scale=TEXT_SCALE * 0.9,
            text_fg=(0.85, 0.85, 0.85, 1),
            text_align=TextNode.ALeft,
            frameColor=LABEL_CLR,
            pos=(-0.38, 0, acc),
            parent=self._right_frame,
        )
        self._radius_entry = DirectEntry(
            text="",
            scale=TEXT_SCALE * 0.9 / 1.25,
            width=ENTRY_W,
            initialText="",
            numLines=1,
            frameColor=(1, 1, 1, 1),
            borderWidth=(0, 0),
            overflow=True,
            text_fg=(0, 0, 0, 1),
            pos=(-0.02, 0, acc),
            parent=self._right_frame,
        )
        self._radius_entry.enterText("0.0")
        # Toggle collision node preview
        acc -= TEXT_SCALE * 2.5
        _make_btn("Toggle CD Preview", self._toggle_cd_preview,
                  self._right_frame, pos=(0, 0, acc),
                  frame_size=(-1.8, 1.8, -0.3, 0.6))

    # ==================================================================
    # Model loading
    # ==================================================================
    def _on_load_btn(self):
        path = self._path_entry.get().strip()
        self._on_load_model(path)

    def _on_load_model(self, path):
        path = path.strip().strip('"').strip("'")
        if not path:
            self._set_status("Please enter a file path", bad=True)
            return
        if not os.path.isfile(path):
            self._set_status(f"File not found: {os.path.basename(path)}", bad=True)
            return
        try:
            # Remove old model
            if self._model_gm is not None:
                self._model_gm.remove()
                self._model_gm = None
            mdl = mgm.GeometricModel(path)
            mdl.rgba = [0.6, 0.6, 0.6, 0.45]
            mdl.attach_to(self.base)
            self._model_gm = mdl
            self._set_status(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            traceback.print_exc()
            self._set_status(f"Load error: {e}", bad=True)

    # ==================================================================
    # Box management
    # ==================================================================
    def _add_box(self):
        idx = len(self._box_data)
        self._box_data.append({
            'cx': 0.0, 'cy': 0.0, 'cz': 0.0,
            'hx': 0.1, 'hy': 0.1, 'hz': 0.1,
        })
        self._box_visuals.append(None)
        self._refresh_box_visual(idx)
        self._rebuild_box_list_ui()
        self._select_box(idx)
        self._set_status(f"Added Box {idx}")

    def _remove_box(self):
        if self._selected_idx < 0 or self._selected_idx >= len(self._box_data):
            self._set_status("Select a box first", bad=True)
            return
        idx = self._selected_idx
        # Remove visual
        if self._box_visuals[idx] is not None:
            self._box_visuals[idx].remove()
        self._box_data.pop(idx)
        self._box_visuals.pop(idx)
        self._rebuild_box_list_ui()
        # Re-select
        if len(self._box_data) > 0:
            new_sel = min(idx, len(self._box_data) - 1)
            self._select_box(new_sel)
        else:
            self._selected_idx = -1
            self._param_title['text'] = "No box selected"
        # Refresh all visuals (indices may have changed)
        for i in range(len(self._box_data)):
            self._refresh_box_visual(i)
        self._refresh_cd_preview()
        self._set_status(f"Removed Box {idx}")

    def _select_box(self, idx):
        if idx < 0 or idx >= len(self._box_data):
            return
        self._selected_idx = idx
        self._param_title['text'] = f"Box {idx} Parameters"
        d = self._box_data[idx]
        mapping = {
            'center_x': 'cx', 'center_y': 'cy', 'center_z': 'cz',
            'half_x': 'hx', 'half_y': 'hy', 'half_z': 'hz',
        }
        for ui_key, data_key in mapping.items():
            self._param_entries[ui_key].enterText(f"{d[data_key]:.4f}")
        # Highlight selected in list
        for i, btn in enumerate(self._box_buttons):
            if i == idx:
                btn['frameColor'] = (0.3, 0.5, 0.8, 1)
            else:
                btn['frameColor'] = (0.25, 0.25, 0.3, 1)

    def _rebuild_box_list_ui(self):
        # Destroy old buttons
        for btn in self._box_buttons:
            btn.destroy()
        self._box_buttons.clear()
        acc = -0.02
        for i in range(len(self._box_data)):
            clr = BOX_COLORS[i % len(BOX_COLORS)]
            clr_indicator = f"[{'%.0f' % (clr[0] * 255)},{'%.0f' % (clr[1] * 255)},{'%.0f' % (clr[2] * 255)}]"
            btn = DirectButton(
                text=f"Box {i} {clr_indicator}",
                text_scale=0.6,
                text_fg=(1, 1, 1, 1),
                text_pos=(0, -0.1),
                text_align=TextNode.ACenter,
                frameSize=(-3.5, 3.5, -0.35, 0.4),
                frameColor=(0.25, 0.25, 0.3, 1),
                relief=DGG.FLAT,
                scale=(TEXT_SCALE, 1, TEXT_SCALE),
                pos=(0, 0, acc),
                command=self._select_box,
                extraArgs=[i],
                parent=self._box_list_frame,
            )
            self._box_buttons.append(btn)
            acc -= TEXT_SCALE * 1.7

    # ==================================================================
    # Parameter editing
    # ==================================================================
    def _on_param_changed(self, key, text):
        """Called when Enter is pressed inside a parameter entry."""
        self._apply_params()

    def _on_param_focus_out(self, key):
        """Called when an entry loses focus."""
        # We don't auto-apply on focus out to avoid confusion; user clicks Apply
        pass

    def _apply_params(self):
        """Read all entry values and update the selected box."""
        if self._selected_idx < 0 or self._selected_idx >= len(self._box_data):
            self._set_status("Select a box first", bad=True)
            return
        mapping = {
            'center_x': 'cx', 'center_y': 'cy', 'center_z': 'cz',
            'half_x': 'hx', 'half_y': 'hy', 'half_z': 'hz',
        }
        d = self._box_data[self._selected_idx]
        for ui_key, data_key in mapping.items():
            txt = self._param_entries[ui_key].get().strip()
            try:
                d[data_key] = float(txt)
            except ValueError:
                self._set_status(f"Invalid value for {ui_key}: {txt}", bad=True)
                return
        self._refresh_box_visual(self._selected_idx)
        self._refresh_cd_preview()
        self._set_status(f"Box {self._selected_idx} updated")

    # ==================================================================
    # Visualization
    # ==================================================================
    def _refresh_box_visual(self, idx):
        """Redraw the box at *idx* as a semi-transparent GeometricModel."""
        if self._box_visuals[idx] is not None:
            self._box_visuals[idx].remove()
            self._box_visuals[idx] = None
        d = self._box_data[idx]
        hx, hy, hz = abs(d['hx']), abs(d['hy']), abs(d['hz'])
        if hx < 1e-6 or hy < 1e-6 or hz < 1e-6:
            return  # zero-size box, skip
        xyz_lengths = np.array([hx * 2, hy * 2, hz * 2])
        pos = np.array([d['cx'], d['cy'], d['cz']])
        clr = BOX_COLORS[idx % len(BOX_COLORS)]
        box_sgm = mgm.gen_box(xyz_lengths=xyz_lengths, pos=pos,
                              rgb=np.array(clr[:3]), alpha=clr[3])
        box_sgm.attach_to(self.base)
        self._box_visuals[idx] = box_sgm

    def _toggle_cd_preview(self):
        """Toggle showing the actual CollisionNode wireframe preview."""
        if self._box_cd_np is not None:
            self._box_cd_np.removeNode()
            self._box_cd_np = None
            self._set_status("CD preview hidden")
            return
        self._refresh_cd_preview()
        self._set_status("CD preview shown")

    def _refresh_cd_preview(self):
        """Rebuild the Panda3D CollisionNode preview."""
        if self._box_cd_np is not None:
            self._box_cd_np.removeNode()
            self._box_cd_np = None
        if not self._box_data:
            return
        try:
            radius = float(self._radius_entry.get().strip())
        except ValueError:
            radius = 0.0
        cnode = CollisionNode("editor_preview")
        for d in self._box_data:
            box = CollisionBox(
                Point3(d['cx'], d['cy'], d['cz']),
                x=abs(d['hx']) + radius,
                y=abs(d['hy']) + radius,
                z=abs(d['hz']) + radius,
            )
            cnode.addSolid(box)
        self._box_cd_np = self.base.render.attachNewNode(cnode)
        self._box_cd_np.show()

    # ==================================================================
    # Code generation
    # ==================================================================
    def _generate_code(self):
        if not self._box_data:
            self._set_status("No boxes to export", bad=True)
            return
        try:
            radius_val = float(self._radius_entry.get().strip())
        except ValueError:
            radius_val = 0.0
        lines = [
            "    @staticmethod",
            "    def _custom_cdprimitive_fn(name, ex_radius):",
            "        pdcnd = mcm.CollisionNode(name + \"_cnode\")",
        ]
        for i, d in enumerate(self._box_data):
            cx, cy, cz = d['cx'], d['cy'], d['cz']
            hx, hy, hz = d['hx'], d['hy'], d['hz']
            lines.append(
                f"        collision_primitive_c{i} = mcm.CollisionBox("
                f"mcm.Point3({cx}, {cy}, {cz}),")
            lines.append(
                f"              {' ' * len(f'collision_primitive_c{i} = mcm.CollisionBox(')}"
                f"x={hx} + ex_radius, y={hy} + ex_radius, z={hz} + ex_radius)")
            lines.append(
                f"        pdcnd.addSolid(collision_primitive_c{i})")
        lines.append("        cdprim = mcm.NodePath(name + \"_cdprim\")")
        lines.append("        cdprim.attachNewNode(pdcnd)")
        lines.append("        return cdprim")
        code = "\n".join(lines)
        print("\n" + "=" * 70)
        print("Generated _cdprimitive_fn code:")
        print("=" * 70)
        print(code)
        print("=" * 70 + "\n")
        # Try to copy to clipboard
        try:
            import subprocess
            process = subprocess.Popen(
                ['clip'], stdin=subprocess.PIPE, shell=True)
            process.communicate(code.encode('utf-8'))
            self._set_status("Code generated & copied to clipboard!")
        except Exception:
            self._set_status("Code generated — see console output")

    # ==================================================================
    # Helpers
    # ==================================================================
    def _set_status(self, msg, bad=False):
        self._status_label['text'] = msg
        if bad:
            self._status_label['text_fg'] = (1, 0.4, 0.4, 1)
        else:
            self._status_label['text_fg'] = (0.7, 0.9, 0.7, 1)


# ======================================================================
if __name__ == '__main__':
    CollisionBoxEditor()