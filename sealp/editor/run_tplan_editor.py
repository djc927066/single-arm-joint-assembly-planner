"""
Run the Task Plan Editor
=========================

Usage::

    python -m sealp.editor.run_tplan_editor
    python -m sealp.editor.run_tplan_editor chair.asmdef
    python -m sealp.editor.run_tplan_editor chair_plan.tplan
"""

import sys

# TracIK must be imported BEFORE panda3d to avoid native library conflicts
try:
    from trac_ik import TracIK as _TracIK  # noqa: F401
except ImportError:
    pass

from sealp.editor.tplan_editor import TplanEditor


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    editor = TplanEditor(input_file=input_file)
    editor.run()


if __name__ == "__main__":
    main()
