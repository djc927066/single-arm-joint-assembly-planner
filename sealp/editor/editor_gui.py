"""
Reusable DirectGUI Widget Factories — v2
==========================================

Polished, consistent DirectGUI helpers for the Assembly Editor.
Dark-theme with accent colours inspired by Blender / Unreal Editor.
"""

import time
from panda3d.core import TextNode, LPoint3f, LVecBase3f, LVecBase4f
from direct.gui.DirectGui import (
    DirectFrame, DirectLabel, DirectButton, DirectScrolledFrame,
    DirectEntry, DGG,
)

# ══════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ══════════════════════════════════════════════════════════════
# Base tones (dark charcoal)
BG_DARK = (0.14, 0.14, 0.16, 0.96)
BG_MID = (0.18, 0.19, 0.22, 0.96)
BG_LIGHT = (0.22, 0.23, 0.26, 0.94)

# Accent
ACCENT = (0.28, 0.56, 0.92, 1.0)       # blue
ACCENT_DIM = (0.22, 0.42, 0.70, 1.0)
ACCENT_BRIGHT = (0.40, 0.72, 1.0, 1.0)
HIGHLIGHT_YELLOW = (0.95, 0.80, 0.25, 1.0)

# Text
TEXT_PRIMARY = (0.92, 0.93, 0.96, 1.0)
TEXT_SECONDARY = (0.62, 0.64, 0.68, 1.0)
TEXT_ACCENT = ACCENT_BRIGHT
TEXT_WARN = (1.0, 0.55, 0.30, 1.0)
TEXT_OK = (0.35, 0.85, 0.50, 1.0)

# Buttons
BTN_NORMAL = (0.24, 0.25, 0.30, 1.0)
BTN_HOVER = (0.32, 0.34, 0.42, 1.0)
BTN_ACTIVE = (0.28, 0.56, 0.92, 1.0)

# List items
LIST_BG = (0.12, 0.12, 0.14, 1.0)
LIST_SEL = (0.22, 0.42, 0.70, 0.55)

# Entries
ENTRY_BG = (0.10, 0.10, 0.12, 1.0)
ENTRY_FG = TEXT_PRIMARY

# Console
CONSOLE_BG = (0.08, 0.08, 0.10, 0.96)

# Separators
SEP_COLOR = (0.30, 0.32, 0.36, 0.5)

# ══════════════════════════════════════════════════════════════
#  SIZES
# ══════════════════════════════════════════════════════════════
TEXT_SIZE = 0.038
SMALL_TEXT = 0.030
TITLE_SIZE = 0.042
SECTION_SIZE = 0.036
LEFT_MARGIN = 0.018
TOP_MARGIN = 0.012
BOTTOM_MARGIN = 0.008
BTN_HEIGHT = 0.032

CONSOLE_TEXT_SIZE = 0.028


# ══════════════════════════════════════════════════════════════
#  PANEL
# ══════════════════════════════════════════════════════════════
def create_panel(parent, width, height, pos, bg=BG_MID):
    """Create a dark panel frame."""
    frame = DirectFrame(
        frameSize=(0, width, -height, 0),
        frameColor=bg,
        pos=LPoint3f(pos[0], 0, pos[1]),
        parent=parent,
    )
    frame.setTransparency(1)
    return frame


def create_section_header(parent, text, pos, width=None,
                          color=ACCENT_BRIGHT):
    """Create a section header with subtle underline bar."""
    lbl = DirectLabel(
        text=text,
        text_scale=SECTION_SIZE,
        text_fg=color,
        text_align=TextNode.ALeft,
        text_font=None,
        frameColor=(0, 0, 0, 0),
        pos=LPoint3f(pos[0], 0, pos[1]),
        parent=parent,
    )
    # underline bar
    if width:
        DirectFrame(
            frameSize=(0, width, -0.002, 0),
            frameColor=(*color[:3], 0.3),
            pos=LPoint3f(pos[0], 0, pos[1] - SECTION_SIZE * 0.5),
            parent=parent,
        )
    return lbl


# ══════════════════════════════════════════════════════════════
#  LABEL
# ══════════════════════════════════════════════════════════════
def create_label(parent, text, pos, scale=TEXT_SIZE, color=TEXT_PRIMARY,
                 align=TextNode.ALeft, word_wrap=None):
    """Create a text label."""
    kwargs = {}
    if word_wrap:
        kwargs['text_wordwrap'] = word_wrap
    return DirectLabel(
        text=text,
        text_scale=scale,
        text_fg=color,
        text_align=align,
        frameColor=(0, 0, 0, 0),
        pos=LPoint3f(pos[0], 0, pos[1]),
        parent=parent,
        **kwargs,
    )


# ══════════════════════════════════════════════════════════════
#  ENTRY FIELD
# ══════════════════════════════════════════════════════════════
def create_entry(parent, pos, width=0.12, initial="0.000",
                 on_change=None):
    """Create a styled text entry with dark background."""
    entry = DirectEntry(
        text="",
        initialText=initial,
        scale=SMALL_TEXT,
        frameColor=ENTRY_BG,
        text_fg=ENTRY_FG,
        width=int(width / SMALL_TEXT),
        pos=LPoint3f(pos[0], 0, pos[1]),
        parent=parent,
        numLines=1,
        focus=0,
        suppressKeys=1,
    )
    if on_change:
        entry['command'] = on_change
    return entry


# ══════════════════════════════════════════════════════════════
#  BUTTON
# ══════════════════════════════════════════════════════════════
def create_button(parent, text, pos, command=None, width=0.20,
                  extra_args=None, height=0.7, text_color=TEXT_PRIMARY,
                  bg=BTN_NORMAL, hover_bg=BTN_HOVER):
    """Create a styled flat button with hover effect."""
    sz = TEXT_SIZE
    half_w = width / (2 * sz)
    btn = DirectButton(
        text=text,
        text_scale=0.85,
        text_fg=text_color,
        frameSize=(-half_w, half_w, -0.30, height),
        frameColor=bg,
        scale=sz,
        pos=LPoint3f(pos[0], 0, pos[1]),
        parent=parent,
        command=command,
        extraArgs=extra_args or [],
        relief=DGG.FLAT,
    )
    # hover effects
    def _enter(_):
        btn['frameColor'] = hover_bg
    def _exit(_):
        btn['frameColor'] = bg
    btn.bind(DGG.ENTER, _enter)
    btn.bind(DGG.EXIT, _exit)
    return btn


def create_accent_button(parent, text, pos, command=None, width=0.20,
                         extra_args=None):
    """Create a blue accent button for primary actions."""
    return create_button(parent, text, pos, command, width, extra_args,
                         bg=ACCENT_DIM, hover_bg=ACCENT,
                         text_color=(1, 1, 1, 1))


# ══════════════════════════════════════════════════════════════
#  LIST ITEM
# ══════════════════════════════════════════════════════════════
def create_list_item(parent, text, pos, width, on_click=None,
                     item_id=None, selected=False):
    """Create a clickable list item row."""
    bg = LIST_SEL if selected else (0, 0, 0, 0)
    fg = TEXT_ACCENT if selected else TEXT_PRIMARY
    row_h = SMALL_TEXT * 1.6
    btn = DirectButton(
        text=text,
        text_scale=SMALL_TEXT,
        text_fg=fg,
        text_align=TextNode.ALeft,
        frameSize=(0, width * 2, -row_h * 0.4, row_h * 0.6),
        frameColor=bg,
        pos=LPoint3f(pos[0], 0, pos[1]),
        parent=parent,
        relief=DGG.FLAT,
        command=on_click,
        extraArgs=[item_id],
    )
    # hover highlight
    hover_bg = (0.25, 0.27, 0.32, 0.6) if not selected else LIST_SEL
    def _enter(_):
        if not selected:
            btn['frameColor'] = hover_bg
    def _exit(_):
        if not selected:
            btn['frameColor'] = (0, 0, 0, 0)
    btn.bind(DGG.ENTER, _enter)
    btn.bind(DGG.EXIT, _exit)
    return btn


# ══════════════════════════════════════════════════════════════
#  SEPARATOR
# ══════════════════════════════════════════════════════════════
def create_separator(parent, pos, width):
    """Create a thin horizontal separator line."""
    return DirectFrame(
        frameSize=(0, width, -0.001, 0.001),
        frameColor=SEP_COLOR,
        pos=LPoint3f(pos[0], 0, pos[1]),
        parent=parent,
    )


# ══════════════════════════════════════════════════════════════
#  CONSOLE WINDOW
# ══════════════════════════════════════════════════════════════
class ConsoleWindow:
    """Scrollable console log panel (like the TBM ConoleUI).

    Parameters
    ----------
    parent : NodePath
        Parent Panda3D node (typically ``base.a2dBottomCenter``).
    width : float
        Half-width in aspect2d coords.
    height : float
        Height of the console panel.
    max_lines : int
        Maximum number of messages to keep.
    """

    def __init__(self, parent, width=1.0, height=0.28,
                 y_offset=0.0, max_lines=50):
        self._max_lines = max_lines
        self._parent_width = width

        # outer frame
        self._frame = DirectFrame(
            frameSize=(-width, width, 0, height),
            frameColor=CONSOLE_BG,
            pos=LPoint3f(0, 0, y_offset),
            parent=parent,
        )
        self._frame.setTransparency(1)

        # title
        DirectLabel(
            text="Console",
            text_scale=SECTION_SIZE,
            text_fg=ACCENT_BRIGHT,
            text_align=TextNode.ALeft,
            frameColor=(0, 0, 0, 0),
            pos=LPoint3f(-width + LEFT_MARGIN, 0, height - SECTION_SIZE * 1.1),
            parent=self._frame,
        )

        # separator under title
        DirectFrame(
            frameSize=(-width + LEFT_MARGIN, width - LEFT_MARGIN, -0.001, 0.001),
            frameColor=SEP_COLOR,
            pos=LPoint3f(0, 0, height - SECTION_SIZE * 1.4),
            parent=self._frame,
        )

        # scrollable area
        scroll_top = height - SECTION_SIZE * 1.6
        self._scroll = DirectScrolledFrame(
            canvasSize=(-width + LEFT_MARGIN, width - LEFT_MARGIN * 2,
                        0, scroll_top),
            frameSize=(-width + LEFT_MARGIN * 0.5, width - LEFT_MARGIN * 0.5,
                       0.01, scroll_top),
            frameColor=(0, 0, 0, 0),
            pos=LPoint3f(0, 0, 0),
            parent=self._frame,
            scrollBarWidth=0.018,
            autoHideScrollBars=True,
        )
        self._scroll_top = scroll_top
        self._labels = []
        self._leftmost = -width + LEFT_MARGIN * 2
        self._last_message = None

    def log(self, text, color=TEXT_PRIMARY, source="Editor"):
        """Add a timestamped message to the console."""
        ts = time.strftime("%H:%M:%S")
        msg = f"[{ts}] [{source}] {text}"
        # skip duplicate consecutive messages
        if msg == self._last_message:
            return
        self._last_message = msg

        # evict oldest if over limit
        if len(self._labels) >= self._max_lines:
            old = self._labels.pop(0)
            old.destroy()
            # shift remaining labels up
            for lbl in self._labels:
                p = lbl.getPos()
                lbl.setPos(p[0], p[1], p[2] + CONSOLE_TEXT_SIZE * 1.2)

        y = self._scroll_top - CONSOLE_TEXT_SIZE * 1.2 * (len(self._labels) + 1)
        lbl = DirectLabel(
            text=msg,
            text_scale=CONSOLE_TEXT_SIZE,
            text_fg=color,
            text_align=TextNode.ALeft,
            frameColor=(0, 0, 0, 0),
            pos=LPoint3f(self._leftmost, 0, y),
            parent=self._scroll.getCanvas(),
        )
        self._labels.append(lbl)

        # expand canvas
        total_h = CONSOLE_TEXT_SIZE * 1.2 * (len(self._labels) + 1)
        if total_h > self._scroll_top:
            self._scroll['canvasSize'] = (
                self._leftmost,
                self._parent_width - LEFT_MARGIN * 2,
                self._scroll_top - total_h,
                self._scroll_top,
            )
        # auto-scroll to bottom
        self._scroll['verticalScroll_value'] = 1.0

    def log_ok(self, text, source="Editor"):
        self.log(text, color=TEXT_OK, source=source)

    def log_warn(self, text, source="Editor"):
        self.log(text, color=TEXT_WARN, source=source)

    def log_info(self, text, source="Editor"):
        self.log(text, color=TEXT_SECONDARY, source=source)

    def clear(self):
        for lbl in self._labels:
            lbl.destroy()
        self._labels.clear()
        self._last_message = None
