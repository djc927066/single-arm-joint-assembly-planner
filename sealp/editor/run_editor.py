"""
Run the Assembly Sequence Editor
=================================

Usage::

    python -m sealp.editor.run_editor
    python -m sealp.editor.run_editor my_assembly.yaml
"""

import sys
from sealp.editor.assembly_editor import AssemblyEditor


def main():
    seq_file = sys.argv[1] if len(sys.argv) > 1 else None
    editor = AssemblyEditor(sequence_file=seq_file)
    editor.run()


if __name__ == "__main__":
    main()
